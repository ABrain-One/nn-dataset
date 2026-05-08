#!/usr/bin/env python3
"""Migrate clean struct1 backbone SFT files into nn-dataset.

This is a one-shot data migration helper. It copies generated struct1 models
and their full training stats into the file-backed nn-dataset layout so the
SQLite DB can be rebuilt from source files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_STAT_FIELDS = {"uid", "transform", "duration", "accuracy"}
DEFAULT_SOURCE_ROOT = Path("/home/s471802/nn-gpt")


@dataclass(frozen=True)
class Candidate:
    code_path: Path
    stat_path: Path | None
    name: str
    source_kind: str


def uuid4_like(obj: object) -> str:
    compact = re.sub(r"\s", "", str(obj))
    return hashlib.md5(compact.encode()).hexdigest()


def canonicalize_python(code: str) -> str:
    tree = ast.parse(code)
    return ast.unparse(tree).strip() + "\n"


def has_full_stat(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    return any(isinstance(row, dict) and REQUIRED_STAT_FIELDS.issubset(row) for row in rows)


def infer_name_from_json_files(directory: Path, prefix: str) -> str | None:
    name_pattern = re.compile(rf"{re.escape(prefix)}-[0-9a-f]{{32}}")
    for json_path in sorted(directory.glob("*.json")):
        try:
            text = json_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = name_pattern.search(text)
        if match:
            return match.group(0)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("nn", "model", "model_name", "nn_name"):
                value = row.get(key)
                if isinstance(value, str) and value.startswith(prefix):
                    return value
    return None


def stat_candidates_for_name(source_root: Path, code_path: Path, name: str, extra_stat_roots: Iterable[Path]) -> list[Path]:
    stat_dir_name = f"img-classification_cifar-10_acc_{name}"
    roots = [
        source_root / "out/nngpt/new_lemur/stat/train",
        source_root / "out/nngpt/stat/train",
        code_path.parent.parent / "stat/train",
        code_path.parent.parent.parent / "stat/train",
        *extra_stat_roots,
    ]
    return [root / stat_dir_name / "1.json" for root in roots]


def discover_candidates(args: argparse.Namespace) -> tuple[list[Candidate], dict[str, list[str]]]:
    source_root = args.source_root.resolve()
    skipped: dict[str, list[str]] = {
        "missing_stat": [],
        "unnamed_synth": [],
    }
    candidates: list[Candidate] = []

    new_lemur_dir = source_root / "out/nngpt/new_lemur/nn"
    extra_stat_roots = [path.resolve() for path in args.stat_root]
    for code_path in sorted(new_lemur_dir.glob(f"{args.prefix}-*.py")):
        name = code_path.stem
        stat_path = next((p for p in stat_candidates_for_name(source_root, code_path, name, extra_stat_roots) if p.exists()), None)
        if stat_path is None:
            skipped["missing_stat"].append(str(code_path))
            continue
        candidates.append(Candidate(code_path, stat_path, name, "new_lemur"))

    for code_path in sorted(source_root.glob("out/nngpt/llm/epoch/A*/synth_nn/B*/new_nn.py")):
        name = None
        if code_path.parent.name.startswith(args.prefix):
            name = code_path.parent.name
        if name is None:
            name = infer_name_from_json_files(code_path.parent, args.prefix)
        if name is None and args.derive_synth_names:
            name = f"{args.prefix}-{uuid4_like(canonicalize_python(code_path.read_text(encoding='utf-8')))}"
        if name is None:
            skipped["unnamed_synth"].append(str(code_path))
            continue

        stat_path = code_path.parent / "1.json"
        if not stat_path.exists():
            skipped["missing_stat"].append(str(code_path))
            continue
        candidates.append(Candidate(code_path, stat_path, name, "synth_nn"))

    return candidates, skipped


def write_migration(args: argparse.Namespace) -> dict[str, object]:
    repo_root = args.repo_root.resolve()
    nn_dir = repo_root / "ab/nn/nn"
    stat_train_dir = repo_root / "ab/nn/stat/train"
    candidates, skipped = discover_candidates(args)

    report: dict[str, object] = {
        "source_root": str(args.source_root.resolve()),
        "repo_root": str(repo_root),
        "dry_run": args.dry_run,
        "candidates": len(candidates),
        "written": [],
        "skipped": {
            **skipped,
            "ast_parse": [],
            "incomplete_stat": [],
            "name_conflict": [],
            "duplicate_same_content": [],
        },
    }

    seen: dict[str, str] = {}
    for candidate in candidates:
        try:
            canonical_code = canonicalize_python(candidate.code_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report["skipped"]["ast_parse"].append(f"{candidate.code_path}: {exc}")
            continue

        try:
            if candidate.stat_path is None or not has_full_stat(candidate.stat_path):
                report["skipped"]["incomplete_stat"].append(str(candidate.stat_path or candidate.code_path))
                continue
        except Exception as exc:
            report["skipped"]["incomplete_stat"].append(f"{candidate.stat_path}: {exc}")
            continue

        previous = seen.get(candidate.name)
        if previous is not None:
            if previous == canonical_code:
                report["skipped"]["duplicate_same_content"].append(str(candidate.code_path))
            else:
                report["skipped"]["name_conflict"].append(str(candidate.code_path))
            continue
        seen[candidate.name] = canonical_code

        dst_code = nn_dir / f"{candidate.name}.py"
        dst_stat = stat_train_dir / f"img-classification_cifar-10_acc_{candidate.name}" / "1.json"

        if dst_code.exists() and dst_code.read_text(encoding="utf-8") != canonical_code:
            report["skipped"]["name_conflict"].append(str(candidate.code_path))
            continue

        report["written"].append(
            {
                "name": candidate.name,
                "source_kind": candidate.source_kind,
                "code": str(dst_code),
                "stat": str(dst_stat),
            }
        )

        if args.dry_run:
            continue
        dst_code.parent.mkdir(parents=True, exist_ok=True)
        dst_stat.parent.mkdir(parents=True, exist_ok=True)
        dst_code.write_text(canonical_code, encoding="utf-8")
        shutil.copy2(candidate.stat_path, dst_stat)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prefix", default="rl-bb-struct1")
    parser.add_argument("--stat-root", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--derive-synth-names", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    report = write_migration(parse_args())
    skipped = report["skipped"]
    print(
        json.dumps(
            {
                "candidates": report["candidates"],
                "written": len(report["written"]),
                "skipped": {key: len(value) for key, value in skipped.items()},
                "dry_run": report["dry_run"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
