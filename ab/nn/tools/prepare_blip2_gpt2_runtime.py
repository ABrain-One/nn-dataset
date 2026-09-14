"""Export a locally available GPT-2 decoder into a BLIP-2 portable bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from ab.nn.captioning.blip2.environment import validate_environment

validate_environment()

from transformers import AutoModelForCausalLM, GPT2TokenizerFast

from ab.nn.captioning.blip2.contract import (
    RUNTIME_DIR_NAME,
    atomic_json,
    read_manifest,
    resolve_cache_dir,
    sha256_file,
    validate_runtime,
)
from ab.nn.captioning.blip2.gpt2 import (
    GPT2_DECODER_DIR_NAME,
    GPT2_MODEL_ID,
    GPT2_TOKENIZER_DIR_NAME,
)


def _runtime_records(runtime: Path):
    return [
        {
            "path": str(path.relative_to(runtime)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(runtime.rglob("*"))
        if path.is_file()
    ]


def export(cache_dir: str | Path, source: str = GPT2_MODEL_ID) -> Path:
    root = resolve_cache_dir(cache_dir)
    manifest = read_manifest(root)
    runtime = validate_runtime(root, manifest)
    decoder = runtime / GPT2_DECODER_DIR_NAME
    tokenizer = runtime / GPT2_TOKENIZER_DIR_NAME

    if not (decoder / "config.json").is_file():
        try:
            model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(source, local_files_only=False)
        model.save_pretrained(decoder, safe_serialization=True)
        del model
    if not (tokenizer / "tokenizer_config.json").is_file():
        try:
            value = GPT2TokenizerFast.from_pretrained(source, local_files_only=True)
        except Exception:
            value = GPT2TokenizerFast.from_pretrained(source, local_files_only=False)
        value.save_pretrained(tokenizer)

    records = _runtime_records(runtime)
    manifest["runtime"] = {"complete": True, "files": records}
    manifest["gpt2_runtime"] = {
        "model_id": GPT2_MODEL_ID,
        "decoder": GPT2_DECODER_DIR_NAME,
        "tokenizer": GPT2_TOKENIZER_DIR_NAME,
    }
    atomic_json(root / "manifest.json", manifest)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--source",
        default=GPT2_MODEL_ID,
        help="Existing local Hugging Face snapshot directory or model identifier already cached locally.",
    )
    args = parser.parse_args()
    root = export(args.cache_dir, args.source)
    print(f"Portable GPT-2 runtime ready: {root / RUNTIME_DIR_NAME}")


if __name__ == "__main__":
    main()
