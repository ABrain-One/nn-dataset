"""Portable GPT-2 decoder helpers for the cached BLIP-2 experiment."""

from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path

import torch

from .contract import CacheError, read_manifest, resolve_cache_dir, validate_runtime

GPT2_MODEL_ID = "openai-community/gpt2"
# The snapshot already used locally; pinning does not upgrade the model.
GPT2_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_VOCAB_SIZE = 50_257
GPT2_DECODER_DIR_NAME = "gpt2-decoder"


@lru_cache(maxsize=None)
def tokenizer():
    """Resolve the pinned GPT-2 tokenizer as a regular HF dependency."""
    from transformers import AutoTokenizer

    value = AutoTokenizer.from_pretrained(
        GPT2_MODEL_ID,
        revision=GPT2_MODEL_REVISION,
        use_fast=True,
    )
    value.pad_token = value.pad_token or value.eos_token
    probe = value("a cat", add_special_tokens=False).input_ids
    if not probe:
        raise CacheError(
            "The pinned GPT-2 tokenizer produced no token IDs; report this "
            "Transformers/tokenizer contract failure upstream."
        )
    if len(value) != GPT2_VOCAB_SIZE:
        raise CacheError(
            "The pinned GPT-2 tokenizer has an incompatible vocabulary: "
            f"expected {GPT2_VOCAB_SIZE}, found {len(value)}."
        )
    return value


def gpt2_decoder_path(cache_dir: str | Path | None = None) -> Path:
    root = resolve_cache_dir(cache_dir)
    manifest = read_manifest(root)
    runtime = validate_runtime(root, manifest)
    record = manifest.get("gpt2_runtime")
    if record is None:
        # GPT-2 is needed only for this model, including with an OPT-only cache.
        from ab.nn.util.captioning.tools.prepare_blip2_gpt2_runtime import export
        export(root)
        manifest = read_manifest(root)
        runtime = validate_runtime(root, manifest)
        record = manifest.get("gpt2_runtime")
    if not isinstance(record, dict) or record.get("model_id") not in {"gpt2", GPT2_MODEL_ID}:
        raise CacheError("Portable GPT-2 runtime has an incompatible model identifier.")
    decoder = runtime / GPT2_DECODER_DIR_NAME
    if not (decoder / "config.json").is_file():
        raise CacheError("Portable GPT-2 runtime is incomplete.")
    return decoder


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
    value = tokenizer()
    token_rows = []
    for text in texts:
        try:
            ids = value.encode(
                text, add_special_tokens=False, truncation=True, max_length=50
            )
        except Exception as error:
            raise CacheError("GPT-2 tokenizer could not encode a caption reference.") from error
        if not ids:
            raise CacheError("GPT-2 tokenizer produced an empty caption reference.")
        if any(not 0 <= int(token) < GPT2_VOCAB_SIZE for token in ids):
            raise CacheError("GPT-2 tokenizer produced an out-of-range token ID.")
        token_rows.append(torch.tensor(ids, dtype=torch.long))

    width = max(map(len, token_rows))

    labels = torch.full(
        (len(batch), count, width), -100, dtype=torch.long
    )
    offset = 0
    for sample, values in enumerate(references):
        size = len(values)
        for position, token_ids in enumerate(token_rows[offset:offset + size]):
            labels[sample, position, :len(token_ids)] = token_ids
        offset += size
    # Keep BLIP token IDs out of the raw COCO vocabulary shared by legacy
    # caption models. ContextVar also prevents concurrent executions in
    # separate contexts from replacing each other's decoder.
    from .context import select_tokenizer

    select_tokenizer(value)
    return features, labels


def collator(cache_dir: str | Path):
    return partial(collate_cached_gpt2_captions, cache_dir=resolve_cache_dir(cache_dir))
