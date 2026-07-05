"""
Shared state that flows through every node in the graph.
Each node reads from this and returns a dict of fields to update.
"""

from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    repo_path: str            # path to the repo being analyzed
    file_tree: str            # text tree of the repo structure
    stack_info: str           # detected tech stack (languages, frameworks, package managers)
    dependencies: str         # raw dependency list pulled from manifest files
    manifest_files: list      # actual manifest filenames found (e.g. ["requirements.txt"])
    important_files: List[str]   # filepaths picked for deep reading
    file_contents: Dict[str, str]  # {filepath: content} for important files
    structure_summary: str    # LLM output: architecture pattern, folder purposes
    risk_notes: str           # LLM output: risky/undocumented areas found
    final_report: str         # the finished markdown report