# Codebase Onboarding Agent (Beta)

An AI agent that reads any codebase and generates an onboarding report:
architecture overview, tech stack, risk areas, and suggested first-week tasks.

Built for freelancers and developers who inherit unfamiliar codebases and
waste hours figuring out where things are and what's fragile.

## How it works

A linear [LangGraph](https://github.com/langchain-ai/langgraph) agent with 4 steps:

1. **Scan repo** — walks the file tree, detects the tech stack, picks the most important files (no LLM call, pure filesystem logic)
2. **Analyze structure** — LLM reasons about architecture pattern and folder purposes
3. **Deep read** — LLM reads key files and flags risky/undocumented areas, marking each as "confirmed" or "inferred"
4. **Generate report** — LLM combines everything into one clean markdown report

Runs on [Groq](https://console.groq.com) (Llama 3.3 70B) for fast, cheap inference.

## Setup

```bash
# 1. Clone and enter the project
cd codebase-onboarding-agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env
# then edit .env and paste your key from console.groq.com/keys
```

## Usage

```bash
python main.py --repo /path/to/some/project --output report.md
```

Example:
```bash
python main.py --repo ~/projects/old-client-app --output onboarding.md
```

The report is saved as a markdown file you can open directly or share with
your team.

## Current limitations (beta)

- Reads a fixed set of important files (entry points + largest files) — very
  large or unusually structured repos may need manual follow-up
- No self-correction loop yet — it's a single pass, not iterative
- Tested primarily on JS/Node and Python projects; other stacks work but are
  less tuned
- Model name and temperature are hardcoded in `agent/graph.py` for now — no
  config file yet
- No retry/timeout handling around LLM calls — a hung request will hang the
  whole run
- On rare occasions the model may state something as "(confirmed)" that it
  actually inferred (e.g. guessing a file's contents from convention rather
  than reading it) — reports should still be spot-checked, not treated as
  ground truth

## Feedback

This is an early beta. If you try it on a real project, I'd genuinely like
to know: did it save you time, and what did it get wrong or miss?