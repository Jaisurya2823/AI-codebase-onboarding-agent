"""
CLI entry point.

Usage:
    python main.py --repo /path/to/some/project --output report.md
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # pulls GROQ_API_KEY from .env if present

from agent.graph import build_graph  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="AI agent that reads a codebase and generates an onboarding report."
    )
    parser.add_argument("--repo", required=True, help="Path to the repository to analyze")
    parser.add_argument("--output", default="onboarding_report.md", help="Output markdown file path")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    if not os.path.isdir(repo_path):
        print(f"Error: '{repo_path}' is not a valid directory.")
        sys.exit(1)

    print(f"Analyzing repo: {repo_path}")
    print("Step 1/4: Scanning repo structure...")

    graph = build_graph()

    initial_state = {
        "repo_path": repo_path,
        "file_tree": "",
        "stack_info": "",
        "dependencies": "",
        "manifest_files": [],
        "important_files": [],
        "file_contents": {},
        "structure_summary": "",
        "risk_notes": "",
        "final_report": "",
    }

    # Stream so the user sees progress node-by-node instead of waiting blind.
    final_state = None
    step_labels = {
        "scan_repo": "Step 1/4: Scanned repo structure",
        "analyze_structure": "Step 2/4: Analyzed architecture",
        "deep_read": "Step 3/4: Reviewed key files for risks",
        "generate_report": "Step 4/4: Generated report",
    }

    for step_output in graph.stream(initial_state):
        node_name = list(step_output.keys())[0]
        print(step_labels.get(node_name, f"Ran: {node_name}"))
        final_state = step_output[node_name]

    if final_state is None or "final_report" not in final_state:
        print("Something went wrong - no report was generated.")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])

    print(f"\nDone. Report saved to: {args.output}")


if __name__ == "__main__":
    main()