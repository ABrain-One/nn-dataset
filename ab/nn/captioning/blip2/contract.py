"""Portable on-disk contract for cached BLIP-2 Q-Former features.

This module intentionally uses only the Python standard library so a cache can
be inspected before importing PyTorch, CUDA, or Transformers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_VERSION = 1
MODEL_ID = "Salesforce/blip2-opt-2.7b-coco"
MODEL_REVISION = "f38cc874b35f3c5a3048b44cd6adae46ca5b2df2"
LANGUAGE_MODEL_ID = "facebook/opt-2.7b"
OPT_VOCAB_SIZE = 50272
FEATURE_SHAPE = (32, 768)
FEATURE_DTYPE = "float16"
MANIFEST_NAME = "manifest.json"
PROJECTION_NAME = "language_projection.pt"
PROJECTION_PATH_ENV = "BLIP2_PROJECTION_PATH"
RUNTIME_DIR_NAME = "runtime"
OPT_DIR_NAME = "opt-decoder"
OPT_TOKENIZER_DIR_NAME = "opt-tokenizer"
SPLITS = frozenset({"train", "val"})


class CacheError(RuntimeError):
    """The cache is missing, incomplete, or incompatible."""


def resolve_cache_dir(value: str | os.PathLike[str] | None = None) -> Path:
    configured = value or os.environ.get("BLIP2_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parents[4]
    return (root / "out" / "blip2-coco-cache-v1").resolve()


def resolve_projection_path(
    cache_dir: Path,
    value: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a projection portably, relative to its cache bundle by default."""
    configured = value or os.environ.get(PROJECTION_PATH_ENV)
    if not configured:
        return (cache_dir / PROJECTION_NAME).resolve()
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = cache_dir / path
    return path.resolve()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / MANIFEST_NAME
    if not path.is_file():
        raise CacheError(
            f"BLIP-2 cache manifest is missing: {path}. Build it with "
            "`python -m ab.nn.tools.build_blip2_cached --help`."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheError(f"Cannot read BLIP-2 cache manifest: {path}") from error
    if value.get("cache_version") != CACHE_VERSION:
        raise CacheError("Unsupported BLIP-2 cache version.")
    if value.get("model_id") != MODEL_ID:
        raise CacheError("Cache was built with a different BLIP-2 checkpoint.")
    if value.get("model_revision") != MODEL_REVISION:
        raise CacheError("Cache was built with a different BLIP-2 revision.")
    if tuple(value.get("feature_shape", ())) != FEATURE_SHAPE:
        raise CacheError("Cache feature shape is incompatible.")
    return value


def validate_runtime(cache_dir: Path, manifest: dict[str, Any]) -> Path:
    """Validate every file in the portable offline decoder bundle."""
    runtime_dir = cache_dir / RUNTIME_DIR_NAME
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("complete"):
        raise CacheError(
            "Portable BLIP-2 runtime is missing. Re-run the cache builder to "
            "export the frozen decoder and OPT tokenizer."
        )
    files = runtime.get("files")
    if not isinstance(files, list) or not files:
        raise CacheError("Portable BLIP-2 runtime has no file manifest.")
    for record in files:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise CacheError(f"Unsafe runtime path: {relative}")
        path = runtime_dir / relative
        if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise CacheError(f"Missing or truncated runtime file: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise CacheError(f"Runtime checksum mismatch: {path}")
    return runtime_dir


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
