"""
Cached BLIP-2 Feature Transform for NAS Image Captioning Pipeline.

This transform activates Caption.py cached feature mode by returning None from
transform(). The dataset returns precomputed BLIP-2/Q-Former features with shape:
    (32, 768)

Important:
- Uses only existing dependencies.
- Uses module-level shared cache so train/val datasets do not load duplicate
  5GB+ tensors into RAM.
- Keeps the frozen BLIP-2 backbone out of the training loop, which is necessary
  for the 20-minute/epoch target.
"""

import os
import torch
from torch.utils.data import Dataset

_DEFAULT_CACHE_DIR = os.environ.get(
    "BLIP2_CACHE_DIR",
    "/home/ghaffar/nn-gpt/out/nngpt/cache",
)

# Module-level shared cache:
# key = real cache dir
# value = {"features": Tensor[N, 32, 768] float16 CPU, "labels": list}
_SHARED_CACHE = {}


def _default_train_shards():
    return [
        "coco_train_499.pt",
        "coco_train_999.pt",
        "coco_train_1499.pt",
        "coco_train_1999.pt",
        "coco_train_2499.pt",
        "coco_train_2999.pt",
        "coco_train_3499.pt",
        "coco_train_final.pt",
    ]


def _discover_shards(cache_dir: str, split: str):
    """
    Prefer split-specific shards if they exist. For this project, train shards are
    known to exist. If val shards are absent, we reuse the same train cache and
    Caption.py later subsets validation to 500 samples.
    """
    prefix = f"coco_{split}_"
    if os.path.isdir(cache_dir):
        files = sorted(
            f for f in os.listdir(cache_dir)
            if f.startswith(prefix) and f.endswith(".pt")
        )
        if files:
            return files

    return _default_train_shards()


def _load_shared_cache(cache_dir: str):
    """
    Load BLIP-2 feature shards once per Python process and reuse them for train
    and val dataset objects. This avoids duplicate RAM usage and prevents the
    process from being killed.
    """
    real_dir = os.path.realpath(cache_dir)

    if real_dir in _SHARED_CACHE:
        print(f"Reusing shared BLIP-2 cache from: {real_dir}")
        return _SHARED_CACHE[real_dir]

    shard_files = _discover_shards(real_dir, split="train")

    print("Pre-loading BLIP-2 cache into shared memory once...")
    print(f"Cache dir: {real_dir}")

    all_features = []
    all_labels = []

    for sf in shard_files:
        shard_path = os.path.join(real_dir, sf)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing BLIP-2 cache shard: {shard_path}")

        data = torch.load(shard_path, weights_only=False, map_location="cpu")

        if "features" not in data or "labels" not in data:
            raise KeyError(f"Shard must contain 'features' and 'labels': {shard_path}")

        all_features.append(data["features"].half())
        all_labels.extend(data["labels"])

        # Help Python release temporary dict references quickly.
        del data
        
        # Prevent OOM by stopping early if we reached the train limit
        limit = int(os.environ.get("NN_TRAIN_LIMIT", 0))
        if limit > 0 and len(all_labels) >= (limit + 500):
            print(f"Reached limit {limit+500}. Stopping shard load to save RAM.")
            break

    features = torch.cat(all_features, dim=0).contiguous()
    labels = all_labels

    _SHARED_CACHE[real_dir] = {
        "features": features,
        "labels": labels,
    }

    print(f"Cache loaded successfully! Total shape: {features.shape}")
    return _SHARED_CACHE[real_dir]


class CachedBlip2Dataset(Dataset):
    """
    Dataset backed by shared precomputed BLIP-2/Q-Former features.

    __getitem__ returns:
        feature: Tensor[32, 768]
        label: raw caption or list of captions
    """

    def __init__(self, cache_dir: str, split: str = "train"):
        self.cache_dir = os.path.realpath(cache_dir)
        self.split = split

        cache = _load_shared_cache(self.cache_dir)
        self._all_features = cache["features"]
        self._all_labels = cache["labels"]

        # Keep validation lightweight. Caption.py also subsets val to 500, but
        # doing it here prevents accidental full validation use elsewhere.
        if split == "val":
            self._length = min(500, len(self._all_labels))
        else:
            self._length = len(self._all_labels)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        feature = self._all_features[idx].float()
        label = self._all_labels[idx]
        return feature, label


def get_collate_fn():
    """
    Collate cached BLIP-2 features and tokenize captions using GPT-2 tokenizer.

    Returns:
        features: Tensor[B, 32, 768]
        captions: Tensor[B, 1, seq_len]
    """
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch], dim=0)
        raw_captions = [item[1] if isinstance(item[1], (list, tuple)) else [item[1]] for item in batch]
        
        max_caps = max(len(caps) for caps in raw_captions)
        if max_caps == 0:
            max_caps = 1
            
        flat_captions = []
        for caps in raw_captions:
            padded_caps = list(caps) + [""] * (max_caps - len(caps))
            for cap in padded_caps:
                flat_captions.append(str(cap).strip() + tokenizer.eos_token)

        tokens = tokenizer(
            flat_captions,
            padding=True,
            truncation=True,
            max_length=40,
            return_tensors="pt",
        )
        
        batch_size = len(batch)
        seq_len = tokens.input_ids.size(1)
        labels = tokens.input_ids.view(batch_size, max_caps, seq_len)

        return features, labels

    return collate_fn


def transform(norm):
    """
    Returning None tells coco_/Caption.py to use get_dataset() instead of raw
    image transforms.
    """
    return None


def get_dataset(cache_dir: str = None, split: str = "train") -> CachedBlip2Dataset:
    resolved = os.path.realpath(cache_dir or _DEFAULT_CACHE_DIR)
    dataset = CachedBlip2Dataset(resolved, split)
    dataset.collate_fn = get_collate_fn()
    return dataset
