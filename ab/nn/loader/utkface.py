# UTKFace dataset loader: age-stratified train/val/test split + age-balanced sampling metadata

import os
import math
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset
from torchvision import transforms as T
from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev = (0.229, 0.224, 0.225)

MINIMUM_ACCURACY = 0.01
_cache_dir = os.path.join(str(data_dir), 'utkface')


def _age_bin(age: float) -> int:
    # 5-year bins reduce imbalance and preserve age coverage across splits.
    return int(age // 5)


def _stratified_split_indices(ages, train_ratio=0.70, val_ratio=0.10, seed=42):
    rng = random.Random(seed)

    bins = {}
    for idx, age in enumerate(ages):
        bins.setdefault(_age_bin(age), []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for _, idxs in bins.items():
        rng.shuffle(idxs)
        n = len(idxs)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # Ensure each non-trivial bin contributes to all splits when possible.
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        n_test = n - n_train - n_val

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    # Validation/checkpoint selection must not reuse random training augmentation.
    # Match the deployed age model's RGB, bilinear 224x224, ImageNet input.
    evaluation_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(__norm_mean, __norm_dev),
    ])
    os.makedirs(_cache_dir, exist_ok=True)

    dataset = load_dataset(
        "nu-delta/utkface",
        split="train",
        cache_dir=_cache_dir,
    )

    full_dataset = _UTKFace(dataset, transform=transform)
    evaluation_dataset = _UTKFace(full_dataset.dataset, transform=evaluation_transform)

    ages = [float(item['age']) for item in full_dataset.dataset]
    train_idx, val_idx, test_idx = _stratified_split_indices(ages, train_ratio=0.70, val_ratio=0.10, seed=42)

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(evaluation_dataset, val_idx)
    test_ds = Subset(evaluation_dataset, test_idx)

    # Attach held-out test set for final evaluation in your Train.py
    val_ds.held_out_test = test_ds

    # Expose per-sample weights so the trainer can optionally use them later if extended.
    train_ages = [ages[i] for i in train_idx]
    bin_counts = Counter(_age_bin(a) for a in train_ages)
    train_ds.sample_weights = torch.tensor(
        [1.0 / math.sqrt(bin_counts[_age_bin(a)]) for a in train_ages],
        dtype=torch.float32
    )

    return (1,), MINIMUM_ACCURACY, train_ds, val_ds


class _UTKFace(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.transform = transform

        valid_items = []
        for item in hf_dataset:
            age = item.get('age', None)
            img = item.get('image', None)

            if age is None or img is None:
                continue

            try:
                age_val = float(age)
            except (TypeError, ValueError):
                continue

            if 0.0 <= age_val <= 116.0:
                valid_items.append(item)

        self.dataset = valid_items

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        age = float(item['age'])

        if img.mode != 'RGB':
            img = img.convert('RGB')

        x = self.transform(img) if self.transform is not None else self._fallback(img)
        y = torch.tensor([age], dtype=torch.float32)
        return x, y

    @staticmethod
    def _fallback(img):
        import numpy as np

        arr = np.array(img, copy=True)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return x
