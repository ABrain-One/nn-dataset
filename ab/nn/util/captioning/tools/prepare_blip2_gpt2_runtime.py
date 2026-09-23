"""Export a locally available GPT-2 decoder into a BLIP-2 portable bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from transformers import AutoModelForCausalLM, AutoTokenizer

from ab.nn.util.captioning.blip2.contract import (
    CacheError,
    RUNTIME_DIR_NAME,
    atomic_json,
    read_manifest,
    resolve_cache_dir,
    sha256_file,
    validate_runtime,
)
from ab.nn.util.captioning.blip2.gpt2 import (
    GPT2_DECODER_DIR_NAME,
    GPT2_MODEL_ID,
    GPT2_MODEL_REVISION,
    GPT2_TOKENIZER_DIR_NAME,
    load_tokenizer_path,
    validate_gpt2_tokenizer,
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
    with FileLock(str(root / ".build.lock")):
        return _export(root, source)


def _export(root: Path, source: str) -> Path:
    manifest = read_manifest(root)
    runtime = validate_runtime(root, manifest)
    if manifest.get("gpt2_runtime"):
        if manifest["gpt2_runtime"].get("model_id") != GPT2_MODEL_ID:
            raise CacheError("Portable GPT-2 runtime has an incompatible model identifier.")
        tokenizer_dir = runtime / GPT2_TOKENIZER_DIR_NAME
        decoder_dir = runtime / GPT2_DECODER_DIR_NAME
        try:
            load_tokenizer_path(tokenizer_dir)
            if not (decoder_dir / "config.json").is_file():
                raise CacheError("Portable GPT-2 decoder is incomplete.")
            # Never overwrite a semantically validated bundle.
            return root
        except CacheError:
            # Repair a tokenizer/runtime produced by an incompatible backend.
            pass
    options = {"revision": GPT2_MODEL_REVISION} if source == GPT2_MODEL_ID else {}
    # Finish both exports before publishing anything into the runtime bundle.
    with TemporaryDirectory(prefix=".gpt2-export-", dir=root) as temporary:
        stage = Path(temporary)
        try:
            model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, **options)
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(source, local_files_only=False, **options)
        model.save_pretrained(stage / GPT2_DECODER_DIR_NAME, safe_serialization=True)
        del model
        value = None
        errors = []
        for local_only in (True, False):
            try:
                candidate = AutoTokenizer.from_pretrained(
                    source, use_fast=True, local_files_only=local_only, **options
                )
                value = validate_gpt2_tokenizer(candidate)
                break
            except Exception as error:
                errors.append(error)
        if value is None and source == GPT2_MODEL_ID:
            # Transformers can change how AutoTokenizer infers classes from a
            # checkpoint.  The canonical tokenizer.json is a stable serialized
            # backend, so use it directly when class inference is incompatible.
            from huggingface_hub import hf_hub_download
            from transformers import PreTrainedTokenizerFast

            tokenizer_json = hf_hub_download(
                repo_id=GPT2_MODEL_ID,
                filename="tokenizer.json",
                revision=GPT2_MODEL_REVISION,
            )
            value = validate_gpt2_tokenizer(
                PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer_json,
                    bos_token="<|endoftext|>",
                    eos_token="<|endoftext|>",
                    unk_token="<|endoftext|>",
                    pad_token="<|endoftext|>",
                    model_max_length=1024,
                )
            )
        if value is None:
            raise CacheError("Could not prepare a compatible GPT-2 tokenizer.") from errors[-1]
        value.save_pretrained(stage / GPT2_TOKENIZER_DIR_NAME)
        load_tokenizer_path(stage / GPT2_TOKENIZER_DIR_NAME)
        for name in (GPT2_DECODER_DIR_NAME, GPT2_TOKENIZER_DIR_NAME):
            destination = runtime / name
            if destination.exists():
                # Only unregistered remnants of an interrupted export reach here.
                os.replace(destination, stage / (name + ".previous"))
            os.replace(stage / name, destination)

    records = _runtime_records(runtime)
    manifest["runtime"] = {"complete": True, "files": records}
    manifest["gpt2_runtime"] = {
        "model_id": GPT2_MODEL_ID,
        "source": source,
        "model_revision": options.get("revision"),
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
