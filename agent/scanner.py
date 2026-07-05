"""
Deterministic, non-LLM repo scanning helpers.
Keeping these as plain Python (not LLM calls) makes the agent fast and
reliable for the parts that don't need reasoning - we only spend LLM
calls where judgment is actually needed (structure analysis, risk-spotting,
report writing).
"""

import os

# Directories we never want to walk into - noise, not signal.
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".next", "target", ".idea", ".vscode",
    "vendor", ".pytest_cache", "coverage", ".cache",
}

# Manifest files that tell us the stack + dependencies.
MANIFEST_FILES = {
    "package.json": "Node.js / JavaScript",
    "requirements.txt": "Python",
    "pyproject.toml": "Python",
    "Pipfile": "Python",
    "Gemfile": "Ruby",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "composer.json": "PHP",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java/Kotlin (Gradle)",
}

# Common entry-point filenames worth reading in full.
ENTRY_POINT_CANDIDATES = {
    "index.js", "index.ts", "app.js", "app.ts", "server.js", "server.ts",
    "main.py", "app.py", "manage.py", "wsgi.py", "asgi.py",
    "main.go", "main.rs", "Program.cs",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb",
    ".java", ".kt", ".php", ".cs", ".c", ".cpp", ".h",
}

MAX_FILE_READ_CHARS = 20000  # cap per-file content sent to the LLM
MAX_IMPORTANT_FILES = 8      # cap how many files get deep-read


def build_file_tree(repo_path: str, max_entries: int = 400) -> str:
    """Walk the repo and return an indented text tree, skipping noise dirs."""
    lines = []
    count = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        depth = root.replace(repo_path, "").count(os.sep)
        indent = "  " * depth
        rel_root = os.path.relpath(root, repo_path)
        if rel_root != ".":
            lines.append(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            if count >= max_entries:
                lines.append(f"{indent}  ... (truncated, repo has more files)")
                return "\n".join(lines)
            lines.append(f"{indent}  {f}")
            count += 1

    return "\n".join(lines) if lines else "(empty repository)"


def detect_stack(repo_path: str) -> tuple[str, str, set[str]]:
    """
    Find manifest files at the repo root (and one level deep) and return
    (stack_description, raw_dependency_text, actual_manifest_filenames_found).
    """
    stacks_found = []
    dependency_chunks = []
    filenames_found = set()

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        depth = root.replace(repo_path, "").count(os.sep)
        if depth > 1:
            continue  # only check root + one level deep

        for f in files:
            if f in MANIFEST_FILES:
                stacks_found.append(MANIFEST_FILES[f])
                filenames_found.add(f)
                full_path = os.path.join(root, f)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read(MAX_FILE_READ_CHARS)
                    dependency_chunks.append(f"--- {f} ---\n{content}")
                except OSError:
                    pass

    stack_description = ", ".join(sorted(set(stacks_found))) or "Unknown (no manifest file found)"
    dependency_text = "\n\n".join(dependency_chunks) or "No dependency manifest found."

    return stack_description, dependency_text, filenames_found


def strip_hallucinated_files(text: str, actual_manifest_files: set[str]) -> str:
    """
    Code-level safety net: LLMs sometimes mention a common manifest file
    (e.g. pyproject.toml) that isn't actually present, just because it's a
    common pattern in training data. Rather than trust prompting alone to
    prevent this, drop any line that references a manifest filename we did
    NOT actually detect in this repo.
    """
    forbidden = [m for m in MANIFEST_FILES if m not in actual_manifest_files]
    if not forbidden:
        return text

    cleaned_lines = []
    for line in text.split("\n"):
        if any(f in line for f in forbidden):
            continue  # drop the line rather than risk a false claim
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def pick_important_files(repo_path: str) -> list[str]:
    """
    Pick a small set of files worth reading in full:
    known entry points first, then the largest code files as a fallback.
    Returns paths relative to repo_path.
    """
    entry_points = []
    other_code_files = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, repo_path)

            if f in ENTRY_POINT_CANDIDATES:
                entry_points.append(rel_path)
            elif os.path.splitext(f)[1] in CODE_EXTENSIONS:
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                other_code_files.append((rel_path, size))

    # Fill remaining slots with the largest code files (likely the most important).
    other_code_files.sort(key=lambda x: x[1], reverse=True)
    remaining_slots = MAX_IMPORTANT_FILES - len(entry_points)
    fallback = [p for p, _ in other_code_files if p not in entry_points][:max(remaining_slots, 0)]

    return (entry_points + fallback)[:MAX_IMPORTANT_FILES]


def read_files(repo_path: str, relative_paths: list[str]) -> dict[str, str]:
    """Read the given files (relative to repo_path), truncated per-file."""
    contents = {}
    for rel_path in relative_paths:
        full_path = os.path.join(repo_path, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read(MAX_FILE_READ_CHARS + 1)  # +1 to detect if truncation happened
            if len(content) > MAX_FILE_READ_CHARS:
                content = (
                    content[:MAX_FILE_READ_CHARS]
                    + "\n\n[... FILE TRUNCATED HERE - the rest of this file was not shown. "
                    "Do not make confirmed claims about code beyond this point ...]"
                )
            contents[rel_path] = content
        except OSError as e:
            contents[rel_path] = f"(could not read file: {e})"
    return contents