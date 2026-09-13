"""NN-Dataset transform backed by validated BLIP-2/Q-Former cache shards.

``transform(norm)`` returns ``None`` to activate NN-Dataset's cached-captioning
branch. A missing or corrupt split auto-extracts the full COCO split in place
(see ``_auto_build_split`` below) rather than silently substituting training
data for validation data or serving a partial/truncated split.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ab.nn.transform.blip2_cache_contract_v2 import (
    FEATURE_SHAPE,
    GPT2_MODEL_ID,
    GPT2_VOCAB_SIZE,
    normalize_split,
    resolve_cache_dir,
)
from ab.nn.transform.blip2_cache_store_v2 import (
    CacheIntegrityError,
    CacheMissingError,
    CacheStore,
    LoadedShard,
    SplitIndex,
)


def _auto_build_split(cache_dir: Path, split: str) -> None:
    """Extract a full split in-process; safe/idempotent if already valid.

    ``build_cache`` re-checks split validity itself and no-ops when a
    complete, correctly-sized cache already exists (see
    ``build_blip2_cache_v2.build_cache``'s own ``force`` fast-path), so this
    is always called on a cache-miss/integrity-error without extra state.
    """
    from ab.nn.tools.build_blip2_cache_v2 import build_cache
    from ab.nn.util.Const import data_dir

    coco_dir = Path(data_dir) / "coco"
    print(
        f"[BLIP-2 cache v2] '{split}' split under {cache_dir} is missing or "
        f"incomplete — auto-extracting the full COCO {split} split from "
        f"{coco_dir} now. This runs once per machine and can take a while."
    )
    build_cache(
        split=split,
        cache_dir=cache_dir,
        coco_dir=coco_dir,
        batch_size=32,
        shard_samples=16_000,
        num_workers=0,
        device_name="auto",
        force=False,
    )
from ab.nn.util.hf.download_utils import ensure_hf_model


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_TOKENIZER = None


def _max_open_shards() -> int:
    raw = os.environ.get("BLIP2_CACHE_OPEN_SHARDS", "2")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"BLIP2_CACHE_OPEN_SHARDS must be positive, received {raw!r}."
        ) from error
    if value <= 0:
        raise ValueError(
            f"BLIP2_CACHE_OPEN_SHARDS must be positive, received {value}."
        )
    return value


def _get_tokenizer():
    global _TOKENIZER

    if _TOKENIZER is None:
        from transformers import GPT2Tokenizer

        local_path = ensure_hf_model(GPT2_MODEL_ID)
        tokenizer = GPT2Tokenizer.from_pretrained(
            local_path,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise RuntimeError("GPT-2 tokenizer has no pad_token_id.")
        _TOKENIZER = tokenizer

    return _TOKENIZER


class CachedBlip2Dataset(Dataset):
    """Lazy, split-strict view of a BLIP-2 feature cache."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        split: str = "train",
    ):
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.split = normalize_split(split)
        self._store = CacheStore(self.cache_dir)
        self._index: SplitIndex | None = None
        self._open_shards: OrderedDict[int, LoadedShard] = OrderedDict()
        self._max_open_shards = _max_open_shards()
        self._collate_fn = None

        # Do not open the split here. NN-Dataset's legacy Caption.py catches a
        # validation-construction error and falls back to the training split.
        # Deferred opening guarantees that missing val data fails loudly later
        # instead of becoming hidden train/validation leakage.

    @property
    def collate_fn(self):
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._collate_fn = value

    def _split_index(self) -> SplitIndex:
        if self._index is None:
            try:
                self._index = self._store.index(self.split)
            except (CacheMissingError, CacheIntegrityError):
                _auto_build_split(self.cache_dir, self.split)
                self._index = self._store.index(self.split)
        return self._index

    def __len__(self) -> int:
        return self._split_index().view_length

    def _load_shard(self, shard_index: int) -> LoadedShard:
        if shard_index in self._open_shards:
            shard = self._open_shards.pop(shard_index)
            self._open_shards[shard_index] = shard
            return shard

        index = self._split_index()
        record = index.records[shard_index]
        shard = self._store.load_shard(
            self.split,
            record,
            legacy=index.legacy,
        )
        self._open_shards[shard_index] = shard

        while len(self._open_shards) > self._max_open_shards:
            self._open_shards.popitem(last=False)

        return shard

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        if not isinstance(index, int):
            raise TypeError(
                f"Dataset index must be int, received "
                f"{type(index).__name__}."
            )

        shard_index, local_index = self._split_index().locate(index)
        shard = self._load_shard(shard_index)
        feature = shard.features[local_index].to(dtype=torch.float32)
        label = shard.labels[local_index]
        return feature, label

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle live memory maps into DataLoader workers."""
        state = dict(self.__dict__)
        state["_open_shards"] = OrderedDict()
        return state


def get_collate_fn():
    """Return a deterministic GPT-2 caption collator."""
    tokenizer = _get_tokenizer()

    def collate_fn(batch):
        if not batch:
            raise ValueError("Cannot collate an empty cached-caption batch.")

        features = torch.stack([sample[0] for sample in batch], dim=0)
        if features.ndim != 3 or tuple(features.shape[1:]) != FEATURE_SHAPE:
            raise RuntimeError(
                "Cached feature batch must have shape "
                f"(B,{FEATURE_SHAPE[0]},{FEATURE_SHAPE[1]}), received "
                f"{tuple(features.shape)}."
            )
        if features.dtype != torch.float32:
            features = features.float()
        if not torch.isfinite(features).all():
            raise ValueError("Cached feature batch contains NaN or Inf.")

        references_per_sample: list[list[str]] = []
        for _, raw_label in batch:
            if isinstance(raw_label, str):
                references = [raw_label.strip()]
            elif isinstance(raw_label, (list, tuple)):
                references = [
                    caption.strip()
                    for caption in raw_label
                    if isinstance(caption, str) and caption.strip()
                ]
            else:
                raise TypeError(
                    "Cached caption must be text or a sequence of texts, "
                    f"received {type(raw_label).__name__}."
                )

            if not references:
                raise ValueError(
                    "A cached sample contains no valid caption references."
                )
            references_per_sample.append(references)

        reference_count = max(map(len, references_per_sample))
        flat_captions: list[str] = []
        valid_references: list[bool] = []

        for references in references_per_sample:
            for reference_index in range(reference_count):
                is_valid = reference_index < len(references)
                text = references[reference_index] if is_valid else ""
                flat_captions.append(
                    f"{text}{tokenizer.eos_token}" if is_valid else ""
                )
                valid_references.append(is_valid)

        tokens = tokenizer(
            flat_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=60,
            add_special_tokens=False,
        )

        labels = tokens.input_ids.clone()
        labels[tokens.attention_mask == 0] = -100
        valid_mask = torch.tensor(valid_references, dtype=torch.bool)
        labels[~valid_mask] = -100

        batch_size = len(batch)
        sequence_length = int(labels.shape[1])
        labels = labels.view(
            batch_size,
            reference_count,
            sequence_length,
        )
        return features, labels

    return collate_fn


def transform(norm):
    """Activate Caption.py's cached-data branch; normalization is irrelevant."""
    del norm
    return None


def get_dataset(
    split: str = "train",
    cache_dir: str | os.PathLike[str] | None = None,
) -> CachedBlip2Dataset:
    dataset = CachedBlip2Dataset(cache_dir=cache_dir, split=split)
    dataset.collate_fn = get_collate_fn()
    return dataset


def get_vocab_size() -> tuple[int]:
    return (GPT2_VOCAB_SIZE,)