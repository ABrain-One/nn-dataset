#!/usr/bin/env python3
"""
Upload age-estimation ONNX/TFLite artifacts to NN-Dataset Hugging Face repos.

Default mapping:
- ONNX files   -> NN-Dataset/onnx under age-regression_utkface_mae/<model>/
- TFLite files -> NN-Dataset/tflite under int8|fp32/age-regression_utkface_mae_<model>/

The script keeps a local upload history to avoid re-uploading unchanged files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_STATS_DIR = REPO_ROOT / "_work" / "stats"
HISTORY_FILE = WORK_STATS_DIR / "upload_history_age_artifacts.json"

DEFAULT_ONNX_REPO = "NN-Dataset/onnx"
DEFAULT_TFLITE_REPO = "NN-Dataset/tflite"
DEFAULT_TASK_TAG = "age-regression_utkface_mae"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_history(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_history(path: Path, history: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)


def infer_model_name(file_path: Path) -> str:
    name = file_path.stem
    if name in {"best_model", "model", "nn"} and file_path.parent.name:
        return file_path.parent.name
    return name


def classify_tflite_bucket(file_path: Path) -> str:
    stem_lower = file_path.stem.lower()
    path_lower = str(file_path).lower()
    if "int8" in stem_lower or "int8" in path_lower:
        return "int8"
    return "fp32"


def discover_artifacts(extra_paths: Iterable[Path] | None = None) -> List[Path]:
    roots = [
        REPO_ROOT / "age estimation model results" / "latest-final-run" / "models",
        REPO_ROOT / "ab" / "nn" / "stat" / "run" / "age-regression_utkface_mae_MobileAgeNet",
    ]
    if extra_paths:
        roots.extend(extra_paths)

    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.onnx", "*.tflite"):
            files.extend(root.rglob(ext))

    unique_sorted = sorted({p.resolve() for p in files if p.is_file()})
    return unique_sorted


def build_remote_target(
    file_path: Path,
    onnx_repo: str,
    tflite_repo: str,
    task_tag: str,
) -> Tuple[str, str]:
    model_name = infer_model_name(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".onnx":
        repo = onnx_repo
        remote = f"{task_tag}/{model_name}/{file_path.name}"
        return repo, remote

    if suffix == ".tflite":
        repo = tflite_repo
        bucket = classify_tflite_bucket(file_path)
        remote = f"{bucket}/{task_tag}_{model_name}/{file_path.name}"
        return repo, remote

    raise ValueError(f"Unsupported artifact type: {file_path}")


def upload_with_retry(
    api: HfApi,
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    max_retries: int,
) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            return
        except Exception as exc:
            msg = str(exc).lower()
            is_retryable = "429" in msg or "too many requests" in msg
            if attempt < max_retries and is_retryable:
                wait_seconds = min(30 * attempt, 120)
                print(f"[WARN] Rate limit while uploading {local_path.name}; retry in {wait_seconds}s")
                time.sleep(wait_seconds)
                continue
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload age-estimation ONNX/TFLite artifacts to NN-Dataset")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token")
    parser.add_argument("--onnx-repo", default=DEFAULT_ONNX_REPO, help="HF repo for ONNX files")
    parser.add_argument("--tflite-repo", default=DEFAULT_TFLITE_REPO, help="HF repo for TFLite files")
    parser.add_argument("--task-tag", default=DEFAULT_TASK_TAG, help="Task path tag in HF repos")
    parser.add_argument("--max-retries", type=int, default=4, help="Max upload retries")
    parser.add_argument("--force", action="store_true", help="Upload even if already present in local history")
    parser.add_argument("--dry-run", action="store_true", help="Print upload plan without uploading")
    parser.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Additional directory to scan for .onnx/.tflite files (can be repeated)",
    )
    args = parser.parse_args()

    token = args.hf_token.strip()
    if not token and not args.dry_run:
        print("[ERROR] Missing HF token. Provide --hf-token or set HF_TOKEN.")
        return 2

    extra_paths = [Path(p).resolve() for p in args.artifact_path]
    artifacts = discover_artifacts(extra_paths)
    if not artifacts:
        print("[INFO] No ONNX/TFLite age artifacts found to upload.")
        return 0

    history = load_history(HISTORY_FILE)
    api = HfApi(token=token) if token else HfApi()

    uploaded = 0
    skipped = 0

    print(f"[INFO] Found {len(artifacts)} artifact(s).")
    for local_path in artifacts:
        file_hash = sha256_file(local_path)
        file_key = str(local_path.relative_to(REPO_ROOT)).replace("\\", "/")
        repo_id, remote_path = build_remote_target(
            file_path=local_path,
            onnx_repo=args.onnx_repo,
            tflite_repo=args.tflite_repo,
            task_tag=args.task_tag,
        )

        prev = history.get(file_key)
        already_uploaded = (
            prev is not None
            and prev.get("sha256") == file_hash
            and prev.get("repo_id") == repo_id
            and prev.get("path_in_repo") == remote_path
        )

        if already_uploaded and not args.force:
            print(f"[SKIP] {file_key} -> {repo_id}/{remote_path} (unchanged)")
            skipped += 1
            continue

        print(f"[PLAN] {file_key} -> {repo_id}/{remote_path}")
        if not args.dry_run:
            upload_with_retry(api, local_path, repo_id, remote_path, args.max_retries)
            history[file_key] = {
                "sha256": file_hash,
                "repo_id": repo_id,
                "path_in_repo": remote_path,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            uploaded += 1

    if not args.dry_run:
        save_history(HISTORY_FILE, history)

    print(f"[DONE] uploaded={uploaded}, skipped={skipped}, dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
