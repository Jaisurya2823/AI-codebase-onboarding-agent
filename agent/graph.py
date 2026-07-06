"""
The agent itself: a linear LangGraph flow.

scan_repo -> analyze_structure -> deep_read -> generate_report -> END

Beta scope on purpose: linear, no retry loops yet. Each node does one job
and hands off to the next via shared AgentState.
"""

import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.scanner import (
    build_file_tree, detect_stack, pick_important_files, read_files,
    strip_hallucinated_files,
)

# Groq model - openai/gpt-oss-120b is Groq's current recommended flagship:
# stronger reasoning than llama-3.3-70b-versatile (which Groq is deprecating),
# still runs at Groq's fast LPU inference speed.
MODEL_NAME = "openai/gpt-oss-120b"


def get_llm(temperature: float = 0.2) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to a .env file or export it before running."
        )
    return ChatGroq(model=MODEL_NAME, temperature=temperature, api_key=api_key)


# ---------------------------------------------------------------------------
# Node 1: Repo Scanner (no LLM - pure filesystem work)
# ---------------------------------------------------------------------------
def scan_repo_node(state: AgentState) -> dict:
    repo_path = state["repo_path"]

    file_tree = build_file_tree(repo_path)
    stack_description, dependency_text, manifest_files = detect_stack(repo_path)
    important_files = pick_important_files(repo_path)

    return {
        "file_tree": file_tree,
        "stack_info": stack_description,
        "dependencies": dependency_text,
        "manifest_files": list(manifest_files),
        "important_files": important_files,
    }


# ---------------------------------------------------------------------------
# Node 2: Structure Analyzer (LLM call #1)
# ---------------------------------------------------------------------------
def analyze_structure_node(state: AgentState) -> dict:
    llm = get_llm()

    system_prompt = (
        "You are a senior software engineer helping a new developer understand "
        "an unfamiliar codebase. Be concise, concrete, and honest about uncertainty. "
        "CRITICAL RULE: only reference filenames that literally appear in the FILE TREE "
        "or DEPENDENCY MANIFESTS given to you below. Never mention a file, config, or "
        "manifest (e.g. pyproject.toml, Dockerfile, .env) unless it is explicitly listed. "
        "If you're inferring something rather than reading it directly, say 'appears to' "
        "or 'likely' - never state an inference as fact."
    )

    user_prompt = f"""Here is a repository's file structure and detected tech stack.
These are the ONLY files that exist in this repo - do not reference any file
not listed here.

TECH STACK: {state['stack_info']}

FILE TREE (the complete, actual list of files - do not assume others exist):
{state['file_tree']}

DEPENDENCY MANIFESTS (only these files were found - if a common manifest like
pyproject.toml isn't listed here, it does not exist in this repo):
{state['dependencies']}

Based ONLY on the above, write a short analysis covering:
1. Likely architecture pattern (e.g. MVC, monolith, microservices, static site, CLI tool)
2. Purpose of each major top-level folder (only folders that appear in the file tree)
3. Any notably outdated or risky dependencies you recognize, quoting only versions
   actually shown in the manifests above (mark clearly as "(inferred)" if you're not certain)

Keep it under 300 words. Use markdown headers."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    cleaned = strip_hallucinated_files(response.content, set(state.get("manifest_files", [])))
    return {"structure_summary": cleaned}


# ---------------------------------------------------------------------------
# Node 3: Deep File Reader + risk analysis (LLM call #2)
# ---------------------------------------------------------------------------
def deep_read_node(state: AgentState) -> dict:
    file_contents = read_files(state["repo_path"], state["important_files"])

    # If we genuinely have no file content to show the model, don't ask it to
    # guess - that's exactly how generic, made-up risk notes get generated.
    # Be honest about the gap instead.
    if not file_contents or not any(
        content.strip() and not content.startswith("(could not read file")
        for content in file_contents.values()
    ):
        return {
            "file_contents": file_contents,
            "risk_notes": (
                "- No file content could be read for deep analysis - this repo's "
                "file types aren't yet supported by this beta's important-file "
                "detection, so no risk notes could be generated for this run. (confirmed)"
            ),
        }

    llm = get_llm()

    system_prompt = (
        "You are a senior engineer reviewing code before handing it off to a new "
        "developer. Look for things that will genuinely trip someone up: missing "
        "error handling, no comments on complex logic, hardcoded secrets or config, "
        "no tests, unclear naming. Only flag things you can actually see in the code "
        "shown to you - do not reference any file that isn't in the list below, and "
        "do not invent issues that aren't visible in the given content. "
        "EVERY bullet point must end with either '(confirmed)' or '(inferred)' - "
        "no exceptions. 'confirmed' means you saw it directly in the code text below. "
        "'inferred' means you're guessing based on general patterns, not something "
        "you actually read."
    )

    file_list = ", ".join(file_contents.keys()) or "(no files were read)"
    files_block = "\n\n".join(
        f"--- {path} ---\n{content}" for path, content in file_contents.items()
    )

    user_prompt = f"""The ONLY files you have access to are: {file_list}

Do not mention any other file, config, or manifest that isn't in this exact list.

FILE CONTENTS:
{files_block}

List the specific risk areas or things that would confuse a new developer.
Every single bullet point MUST end with "(confirmed)" or "(inferred)" -
this is mandatory, not optional. Keep it under 350 words, use a markdown
bullet list."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    cleaned = strip_hallucinated_files(response.content, set(state.get("manifest_files", [])))

    # Code-level fallback: if the model still dropped a tag on some line,
    # append "(unverified)" so nothing reaches the report silently unlabeled.
    tagged_lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        is_bullet = stripped.startswith(("-", "*"))
        has_tag = "(confirmed)" in line or "(inferred)" in line
        if is_bullet and not has_tag:
            line = f"{line} (unverified)"
        tagged_lines.append(line)
    final_risk_notes = "\n".join(tagged_lines)

    return {"file_contents": file_contents, "risk_notes": final_risk_notes}


# ---------------------------------------------------------------------------
# Node 4: Report Generator (LLM call #3)
# ---------------------------------------------------------------------------
def generate_report_node(state: AgentState) -> dict:
    """
    IMPORTANT: structure_summary and risk_notes are already correct and
    already carry their (confirmed)/(inferred) tags. We do NOT run them
    through another LLM pass here - a prior version did, and the rewrite
    silently dropped the tags. The LLM is only used for the two sections
    that genuinely need fresh synthesis: Overview and First-Week Tasks.
    Everything else is inserted verbatim via Python string formatting.
    """
    llm = get_llm(temperature=0.3)

    system_prompt = (
        "You write clear, concise onboarding notes for developers. "
        "Follow the exact output format requested - no extra commentary."
    )

    user_prompt = f"""Based on this architecture analysis and risk notes, write two things.

ARCHITECTURE ANALYSIS:
{state['structure_summary']}

RISK NOTES:
{state['risk_notes']}

TECH STACK: {state['stack_info']}

1. OVERVIEW: a 2-3 sentence summary of what this project is and does.
2. TASKS: 4-6 concrete first-week tasks for a new developer, as markdown
   bullet points (e.g. "run the test suite locally", "trace how X flows through Y module").

Respond in EXACTLY this format, nothing else before or after:

OVERVIEW:
<your paragraph here>

TASKS:
<your bullet list here>"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    content = response.content

    if "OVERVIEW:" in content and "TASKS:" in content:
        overview = content.split("OVERVIEW:", 1)[1].split("TASKS:", 1)[0].strip()
        tasks = content.split("TASKS:", 1)[1].strip()
    else:
        # Fallback if the model doesn't follow the format - better a rough
        # answer than a crash.
        overview = content.strip()
        tasks = "- (Could not generate task list - see Risk Areas above for where to start.)"

    final_report = f"""# Codebase Onboarding Report

## Overview
{overview}

## Architecture
{state['structure_summary']}

## Tech Stack & Dependencies
**Detected stack:** {state['stack_info']}

## Risk Areas & Things to Watch
{state['risk_notes']}

## Suggested First-Week Tasks
{tasks}
"""

    return {"final_report": final_report}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("scan_repo", scan_repo_node)
    graph.add_node("analyze_structure", analyze_structure_node)
    graph.add_node("deep_read", deep_read_node)
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("scan_repo")
    graph.add_edge("scan_repo", "analyze_structure")
    graph.add_edge("analyze_structure", "deep_read")
    graph.add_edge("deep_read", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()