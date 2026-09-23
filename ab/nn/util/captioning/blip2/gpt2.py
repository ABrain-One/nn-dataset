"""Portable GPT-2 decoder helpers for the cached BLIP-2 experiment."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch

from .contract import CacheError, read_manifest, resolve_cache_dir, validate_runtime

GPT2_MODEL_ID = "gpt2"
# The snapshot already used locally; pinning does not upgrade the model.
GPT2_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_VOCAB_SIZE = 50_257
GPT2_DECODER_DIR_NAME = "gpt2-decoder"
GPT2_TOKENIZER_DIR_NAME = "gpt2-tokenizer"


def validate_gpt2_tokenizer(value: Any) -> Any:
    """Validate behavior required by the cached GPT-2 caption pipeline.

    This deliberately checks capabilities instead of package versions.  In
    particular, Transformers 5 can construct an empty trainable tokenizer from
    files that older releases interpreted as a pretrained tokenizer.
    """
    try:
        vocabulary_size = len(value)
        eos_token_id = value.eos_token_id
        probe = value.encode("A photo of a dog.", add_special_tokens=False)
    except Exception as error:
        raise CacheError("Portable GPT-2 tokenizer cannot encode text.") from error
    if vocabulary_size != GPT2_VOCAB_SIZE:
        raise CacheError(
            "Portable GPT-2 tokenizer has an incompatible vocabulary: "
            f"expected {GPT2_VOCAB_SIZE}, found {vocabulary_size}."
        )
    if eos_token_id is None or not 0 <= int(eos_token_id) < GPT2_VOCAB_SIZE:
        raise CacheError("Portable GPT-2 tokenizer has no compatible EOS token.")
    if not probe or any(not 0 <= int(token) < GPT2_VOCAB_SIZE for token in probe):
        raise CacheError("Portable GPT-2 tokenizer produced an invalid probe encoding.")
    value.pad_token = value.eos_token
    return value


def load_tokenizer_path(path: str | Path):
    """Load a validated GPT-2 tokenizer across Transformers tokenizer backends."""
    from transformers import AutoTokenizer

    root = Path(path)
    errors = []
    try:
        return validate_gpt2_tokenizer(
            AutoTokenizer.from_pretrained(str(root), use_fast=True, local_files_only=True)
        )
    except Exception as error:
        errors.append(error)

    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            # Loading the serialized backend directly avoids version-specific
            # AutoTokenizer class inference while retaining a public API.
            from transformers import PreTrainedTokenizerFast

            return validate_gpt2_tokenizer(
                PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_json),
                    bos_token="<|endoftext|>",
                    eos_token="<|endoftext|>",
                    unk_token="<|endoftext|>",
                    pad_token="<|endoftext|>",
                    model_max_length=1024,
                )
            )
        except Exception as error:
            errors.append(error)
    raise CacheError("Portable GPT-2 tokenizer is empty or incompatible.") from errors[-1]


def gpt2_runtime_paths(cache_dir: str | Path | None = None) -> tuple[Path, Path]:
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
        _, path = gpt2_runtime_paths(root)
        try:
            value = load_tokenizer_path(path)
        except CacheError:
            # A previously published manifest may describe a tokenizer that a
            # newer backend serialized as empty.  Repair only the small GPT-2
            # runtime; COCO features and the OPT runtime remain untouched.
            from ab.nn.util.captioning.tools.prepare_blip2_gpt2_runtime import export

            export(root)
            _, path = gpt2_runtime_paths(root)
            value = load_tokenizer_path(path)
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
