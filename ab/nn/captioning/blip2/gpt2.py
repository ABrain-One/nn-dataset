"""Portable GPT-2 decoder helpers for the cached BLIP-2 experiment."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from .contract import CacheError, RUNTIME_DIR_NAME, read_manifest, resolve_cache_dir, validate_runtime

GPT2_MODEL_ID = "gpt2"
# The snapshot already used locally; pinning does not upgrade the model.
GPT2_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_VOCAB_SIZE = 50_257
GPT2_DECODER_DIR_NAME = "gpt2-decoder"
GPT2_TOKENIZER_DIR_NAME = "gpt2-tokenizer"


def gpt2_runtime_paths(cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    root = resolve_cache_dir(cache_dir)
    manifest = read_manifest(root)
    runtime = validate_runtime(root, manifest)
    record = manifest.get("gpt2_runtime")
    if record is None:
        # GPT-2 is needed only for this model, including with an OPT-only cache.
        from ab.nn.tools.prepare_blip2_gpt2_runtime import export
        export(root)
        manifest = read_manifest(root)
        runtime = validate_runtime(root, manifest)
        record = manifest.get("gpt2_runtime")
    if not isinstance(record, dict) or record.get("model_id") != GPT2_MODEL_ID:
        raise CacheError("Portable GPT-2 runtime has an incompatible model identifier.")
    decoder = runtime / GPT2_DECODER_DIR_NAME
    tokenizer = runtime / GPT2_TOKENIZER_DIR_NAME
    if not (decoder / "config.json").is_file() or not tokenizer.is_dir():
        raise CacheError("Portable GPT-2 runtime is incomplete.")
    return decoder, tokenizer


_TOKENIZERS = {}


def tokenizer(cache_dir: str | Path | None = None):
    root = resolve_cache_dir(cache_dir)
    key = str(root)
    if key not in _TOKENIZERS:
        from transformers import GPT2TokenizerFast

        _, path = gpt2_runtime_paths(root)
        value = GPT2TokenizerFast.from_pretrained(str(path), local_files_only=True)
        value.pad_token = value.eos_token
        _TOKENIZERS[key] = value
    return _TOKENIZERS[key]


def collate_cached_gpt2_captions(batch, *, cache_dir: str | Path):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    features = torch.stack([item[0] for item in batch])
    references = [item[1] for item in batch]
    count = max(len(value) for value in references)
    texts, valid = [], []
    for values in references:
        clean = [str(text).strip() for text in values if str(text).strip()]
        if not clean:
            raise CacheError("A cache sample has no caption.")
        for position in range(count):
            present = position < len(clean)
            texts.append(clean[position] if present else "")
            valid.append(present)
    value = tokenizer(cache_dir)
    encoded = value(texts, padding=True, truncation=True, max_length=50, return_tensors="pt")
    labels = encoded.input_ids
    labels[encoded.attention_mask == 0] = -100
    labels[~torch.tensor(valid, dtype=torch.bool)] = -100
    # Metrics are shared, but the token-ID contract is model-specific.
    from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

    GLOBAL_CAPTION_VOCAB["tokenizer"] = value
    return features, labels.view(len(batch), count, -1)


def collator(cache_dir: str | Path):
    return partial(collate_cached_gpt2_captions, cache_dir=resolve_cache_dir(cache_dir))
