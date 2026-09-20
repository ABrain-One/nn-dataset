"""Portable GPT-2 decoder helpers for the cached BLIP-2 experiment."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import torch

from .contract import CacheError, RUNTIME_DIR_NAME, read_manifest, resolve_cache_dir, validate_runtime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
        from transformers import AutoTokenizer

        _, path = gpt2_runtime_paths(root)
        value = AutoTokenizer.from_pretrained(
            str(path), use_fast=True, local_files_only=True
        )
        value.pad_token = value.eos_token
        _TOKENIZERS[key] = value
    return _TOKENIZERS[key]


def collate_cached_gpt2_captions(batch, *, cache_dir: str | Path):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    features = torch.stack([item[0] for item in batch])
    references = []
    for _, values in batch:
        if not isinstance(values, (list, tuple)):
            raise CacheError("Cache captions must be a list or tuple of strings.")
        clean = [str(text).strip() for text in values if str(text).strip()]
        if not clean:
            raise CacheError("A cache sample has no caption.")
        references.append(clean)

    # Tokenize real captions only.  Passing synthetic empty strings through a
    # tokenizer made the missing-reference sentinel dependent on Transformers
    # internals and caused all--100 rows on newer releases.
    count = max(map(len, references))
    texts = [text for values in references for text in values]
    value = tokenizer(cache_dir)
    encoded = value(
        texts, padding=True, truncation=True, max_length=50, return_tensors="pt"
    )
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    attention_mask = torch.as_tensor(encoded["attention_mask"], dtype=torch.bool)
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise CacheError("GPT-2 tokenizer returned an incompatible caption batch.")
    if len(input_ids) != len(texts) or not attention_mask.any(dim=1).all():
        raise CacheError("GPT-2 tokenizer produced an empty caption reference.")

    labels = torch.full(
        (len(batch), count, input_ids.shape[1]), -100, dtype=torch.long
    )
    offset = 0
    for sample, values in enumerate(references):
        size = len(values)
        token_ids = input_ids[offset:offset + size].clone()
        token_ids[~attention_mask[offset:offset + size]] = -100
        labels[sample, :size] = token_ids
        offset += size
    # Metrics are shared, but the token-ID contract is model-specific.
    from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

    GLOBAL_CAPTION_VOCAB.clear()
    GLOBAL_CAPTION_VOCAB["tokenizer"] = value
    return features, labels


def collator(cache_dir: str | Path):
    return partial(collate_cached_gpt2_captions, cache_dir=resolve_cache_dir(cache_dir))
