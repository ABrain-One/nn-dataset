"""Validated lazy dataset for BLIP-2 cache shards."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from functools import partial
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .contract import (
    FEATURE_SHAPE, MANIFEST_NAME, CacheError, read_manifest, resolve_cache_dir, sha256_file,
)


def _split_needs_full_build(manifest: dict, split: str) -> bool:
    """True if a split is absent, still mid-build, or was extracted with --limit.

    A --limit build (e.g. the README's 16-image smoke cache) sets
    ``complete: true`` for that limited size, so ``complete`` alone can't
    detect it; ``limited`` is the flag ``build_blip2_cached.build()`` sets for
    exactly this case.
    """
    record = manifest.get("splits", {}).get(split)
    if not isinstance(record, dict):
        return True
    return not record.get("complete") or bool(record.get("limited"))


def _auto_build_split(cache_dir: Path, split: str) -> None:
    """Extract the full split in-process, resuming any partial/smoke shards.

    Runs once per machine per split; every later call sees a complete,
    unlimited manifest and returns immediately without importing this.
    """
    from ab.nn.util.Const import data_dir
    from ab.nn.tools.build_blip2_cached import build as build_cache_split

    coco_root = data_dir / "coco"
    print(
        f"[BLIP-2 cache] '{split}' split under {cache_dir} is missing, partial, or a "
        f"limited smoke build — auto-extracting the full COCO {split} split from "
        f"{coco_root} now. This runs once per machine and can take a while."
    )
    build_cache_split(coco_root, cache_dir, split, batch_size=32, shard_size=256, limit=None)


class CachedCaptionDataset(Dataset):
    def __init__(self, split: str, cache_dir: str | None = None, verify: bool = True):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.cache_dir = resolve_cache_dir(cache_dir)

        # Fresh machine: no cache at all yet. Safe to build from scratch.
        if not (self.cache_dir / MANIFEST_NAME).is_file():
            _auto_build_split(self.cache_dir, split)

        # An existing manifest with the wrong contract (model/revision/shape)
        # still fails loudly here — never silently rebuilt over.
        manifest = read_manifest(self.cache_dir)

        if _split_needs_full_build(manifest, split):
            _auto_build_split(self.cache_dir, split)
            manifest = read_manifest(self.cache_dir)

        record = manifest.get("splits", {}).get(split)
        if not isinstance(record, dict) or not record.get("complete"):
            raise CacheError(f"BLIP-2 cache split {split!r} is not complete.")
        self.split = split
        self.shards = list(record.get("shards", []))
        if not self.shards:
            raise CacheError(f"BLIP-2 cache split {split!r} has no shards.")
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            path = self.cache_dir / shard["filename"]
            if not path.is_file() or path.stat().st_size != int(shard["size_bytes"]):
                raise CacheError(f"Missing or truncated cache shard: {path}")
            if verify and sha256_file(path) != shard["sha256"]:
                raise CacheError(f"Checksum mismatch for cache shard: {path}")
            total += int(shard["samples"])
            self.ends.append(total)
        limit_name = f"BLIP2_{split.upper()}_LIMIT"
        raw_limit = os.environ.get(limit_name, "0")
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise ValueError(
                f"{limit_name} must be a non-negative integer, got {raw_limit!r}."
            ) from error
        if limit < 0:
            raise ValueError(f"{limit_name} must be non-negative, got {limit}.")
        self._source_length = total
        self._length = min(total, limit) if limit else total
        self._open: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.collate_fn = partial(collate_cached_captions, cache_dir=self.cache_dir)
        self.num_workers = 0  # mmap + deterministic low-memory behavior

    def __len__(self) -> int:
        return self._length

    def _load(self, index: int) -> dict[str, Any]:
        if index in self._open:
            value = self._open.pop(index)
            self._open[index] = value
            return value
        path = self.cache_dir / self.shards[index]["filename"]
        # PyTorch's mmap loader requires a string filename on supported releases.
        value = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        features = value.get("features")
        captions = value.get("captions")
        if not torch.is_tensor(features) or tuple(features.shape[1:]) != FEATURE_SHAPE:
            raise CacheError(f"Invalid feature tensor in {path}")
        if not isinstance(captions, list) or len(captions) != len(features):
            raise CacheError(f"Invalid captions in {path}")
        self._open[index] = value
        while len(self._open) > 2:
            self._open.popitem(last=False)
        return value

    def __getitem__(self, index: int):
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        shard = self._load(shard_index)
        local = index - start
        return shard["features"][local].float(), shard["captions"][local]


_TOKENIZERS = {}


def _tokenizer(cache_dir: Path):
    key = str(cache_dir)
    if key not in _TOKENIZERS:
        from transformers import AutoTokenizer
        from .contract import OPT_TOKENIZER_DIR_NAME, RUNTIME_DIR_NAME
        path = cache_dir / RUNTIME_DIR_NAME / OPT_TOKENIZER_DIR_NAME
        if not path.is_dir():
            raise CacheError(f"Offline OPT tokenizer is missing: {path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), use_fast=False, local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZERS[key] = tokenizer
    # Restore this even on a cache hit: another model may have set GPT-2.
    from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
    GLOBAL_CAPTION_VOCAB["tokenizer"] = _TOKENIZERS[key]
    return _TOKENIZERS[key]


def collate_cached_captions(batch, *, cache_dir: Path):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    features = torch.stack([item[0] for item in batch])
    references = [item[1] for item in batch]
    count = max(len(value) for value in references)
    flattened = []
    valid = []
    for values in references:
        clean = [str(text).strip() for text in values if str(text).strip()]
        if not clean:
            raise CacheError("A cache sample has no caption.")
        for position in range(count):
            present = position < len(clean)
            flattened.append(clean[position] if present else "")
            valid.append(present)
    tokens = _tokenizer(cache_dir)(flattened, padding=True, truncation=True, max_length=50, return_tensors="pt")
    labels = tokens.input_ids
    labels[tokens.attention_mask == 0] = -100
    labels[~torch.tensor(valid, dtype=torch.bool)] = -100
    return features, labels.view(len(batch), count, -1)
