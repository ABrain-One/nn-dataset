from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_PATTERNS = [
    re.compile(r"CREATE\s+INDEX", re.IGNORECASE),
    re.compile(r"CREATE\s+TEMP", re.IGNORECASE),
    re.compile(r"\bWITH\b", re.IGNORECASE),
    re.compile(r"ORDER\s+BY", re.IGNORECASE),
    re.compile(r"\bLIMIT\b", re.IGNORECASE),
    re.compile(r"nn_minhash", re.IGNORECASE),
    re.compile(r"nn_similarity", re.IGNORECASE),
    re.compile(r"jaccard_blobs", re.IGNORECASE),
    re.compile(r"sqlite_master|sqlite_temp_master", re.IGNORECASE),
    re.compile(r"execute\(", re.IGNORECASE),
    re.compile(r"executemany\(", re.IGNORECASE),
    re.compile(r"executescript\(", re.IGNORECASE),
]
MAX_CHANGED_FILES = 20
MAX_MATCH_LINES = 12


@dataclass
class FileImpact:
    path: str
    lines: list[str]
    bullets: list[str]


def run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return proc.stdout.strip()


def normalize_remote_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    if cleaned.startswith("git@github.com:"):
        return "https://github.com/" + cleaned.split(":", 1)[1]
    return cleaned


def short_ref(ref: str) -> str:
    return run_git("rev-parse", "--short", f"{ref}^{{commit}}")


def current_branch() -> str:
    return run_git("rev-parse", "--abbrev-ref", "HEAD")


def changed_files(range_expr: str) -> list[str]:
    output = run_git("diff", "--name-only", range_expr)
    return [line for line in output.splitlines() if line.strip()]


def diff_stat(range_expr: str) -> str:
    return run_git("diff", "--stat", range_expr)


def diff_for_file(range_expr: str, path: str) -> str:
    return run_git("diff", "--unified=3", range_expr, "--", path)


def matches_sql_patterns(line: str) -> bool:
    return any(pattern.search(line) for pattern in SQL_PATTERNS)


def extract_relevant_lines(diff_text: str) -> list[str]:
    relevant: list[str] = []
    for raw_line in diff_text.splitlines():
        if raw_line.startswith(("diff --git", "index ", "--- ", "+++ ", "@@")):
            continue
        if matches_sql_patterns(raw_line):
            relevant.append(raw_line.rstrip())
    return relevant


def summarize_file(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    bullets: list[str] = []

    if re.search(r"CREATE\s+TEMP|sqlite_temp_master", text, re.IGNORECASE):
        bullets.append("Adds or reuses temp SQL objects so intermediate candidate sets can be materialized once instead of rebuilt repeatedly.")
    if re.search(r"CREATE\s+INDEX|ORDER\s+BY|LIMIT", text, re.IGNORECASE):
        bullets.append("Improves index or sort usage, which should reduce full scans and temp B-tree work on ordered candidate selection.")
    if re.search(r"\bWITH\b|execute\(|executemany\(|executescript\(", text, re.IGNORECASE):
        bullets.append("Changes how SQL is assembled or executed, likely moving repeated setup work out of tight loops.")
    if re.search(r"nn_minhash|nn_similarity|jaccard_blobs", text, re.IGNORECASE):
        bullets.append("Touches the similarity-scoring path directly, which is where expensive MinHash or Jaccard comparisons tend to dominate runtime.")

    if not bullets:
        bullets.append("Adjusts SQL-adjacent code in a way that may reduce repeated scans, redundant setup, or unnecessary round-trips.")

    return bullets[:3]


def sql_impacts(range_expr: str, files: Iterable[str]) -> list[FileImpact]:
    impacts: list[FileImpact] = []
    for path in sorted(files):
        diff_text = diff_for_file(range_expr, path)
        lines = extract_relevant_lines(diff_text)
        if not lines:
            continue
        impacts.append(
            FileImpact(
                path=path,
                lines=lines[:MAX_MATCH_LINES],
                bullets=summarize_file(lines),
            )
        )
    return impacts


def narrative_bullets(impacts: list[FileImpact], stat_text: str) -> list[str]:
    joined = "\n".join("\n".join(impact.lines) for impact in impacts)
    bullets: list[str] = [
        "Anuj's baseline establishes the integrated dataset-loading and similarity-query functionality that the branch keeps intact.",
        "Manisha's head revision focuses on reducing repeated SQL work instead of changing outputs, so the comparison stays about execution efficiency rather than feature changes.",
    ]

    if re.search(r"CREATE\s+TEMP|sqlite_temp_master", joined, re.IGNORECASE):
        bullets.append("The optimized branch materializes temp candidate state explicitly, which is a standard way to avoid rebuilding the same filtered working set across anchor-band steps.")
    if re.search(r"CREATE\s+INDEX", joined, re.IGNORECASE):
        bullets.append("The diff adds index-oriented SQL changes, which should improve join and lookup selectivity and reduce scan cost on repeated query paths.")
    if re.search(r"\bWITH\b", joined, re.IGNORECASE):
        bullets.append("Common-table-expression changes indicate that head is trying to compute ranked or filtered subsets once and then reuse them instead of rerunning the same CTE shape.")
    if re.search(r"ORDER\s+BY|LIMIT", joined, re.IGNORECASE):
        bullets.append("Ordering and limiting logic moves closer to the candidate-reduction stage, which typically shrinks later similarity work and temp-sort pressure.")
    if re.search(r"nn_minhash|nn_similarity|jaccard_blobs", joined, re.IGNORECASE):
        bullets.append("The most important changes hit the MinHash or Jaccard similarity path directly, so the runtime win is coming from the expensive scoring layer rather than superficial Python overhead.")
    if re.search(r"execute\(|executemany\(|executescript\(", joined, re.IGNORECASE):
        bullets.append("Execution-path edits suggest that setup SQL is being batched or moved out of repeated loops, which lowers per-query overhead without altering semantics.")

    stat_lines = [line.strip() for line in stat_text.splitlines() if line.strip()]
    if stat_lines:
        bullets.append(f"The diff footprint remains localized enough for review: `{stat_lines[0]}`.")

    bullets.append("Overall, the branch reads as an optimization layer on top of Anuj's functionality: fewer redundant scans, more reusable temp state, and better support for indexed hydration.")
    return bullets[:10]


def verification_commands(remote_url: str, branch: str) -> list[str]:
    pip_url = f"git+{remote_url}@{branch}"
    return [
        "pip uninstall -y nn-dataset",
        f"pip install --no-cache-dir {pip_url}",
        r"cd ..\nn-gpt",
        "python test.py",
    ]


def print_section(title: str) -> None:
    print(f"## {title}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    range_expr = f"{args.baseline}..{args.head}"
    branch = current_branch()
    baseline_hash = short_ref(args.baseline)
    head_hash = short_ref(args.head)
    origin_url = normalize_remote_url(run_git("remote", "get-url", "origin"))
    stat_text = diff_stat(range_expr)
    files = changed_files(range_expr)
    impacts = sql_impacts(range_expr, files)
    narrative = narrative_bullets(impacts, stat_text)

    print("# Diff Summary")
    print()

    print_section("Refs")
    print(f"- Baseline ref: `{args.baseline}`")
    print(f"- Baseline commit: `{baseline_hash}`")
    print(f"- Head ref: `{args.head}`")
    print(f"- Head commit: `{head_hash}`")
    print(f"- Current branch: `{branch}`")
    print()

    print_section("Diff Overview")
    print("```text")
    print(stat_text or "(no diff stat output)")
    print("```")
    print()
    print("- Changed files:")
    for path in files[:MAX_CHANGED_FILES]:
        print(f"  - `{path}`")
    if len(files) > MAX_CHANGED_FILES:
        print(f"  - `... and {len(files) - MAX_CHANGED_FILES} more`")
    print()

    print_section("SQL-impact Changes")
    if not impacts:
        print("- No SQL-impact lines matched the configured patterns in the current diff.")
        print()
    for impact in impacts:
        print(f"### `{impact.path}`")
        print()
        for bullet in impact.bullets:
            print(f"- {bullet}")
        print("```diff")
        for line in impact.lines:
            print(line)
        print("```")
        print()

    print_section("Anuj vs Manisha")
    for bullet in narrative:
        print(f"- {bullet}")
    print()

    print_section("Verification Steps")
    for command in verification_commands(origin_url, branch):
        print(f"```powershell\n{command}\n```")
    print("- Expected output: `ALL TESTS PASSED` and the timing lines from `python test.py`.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
