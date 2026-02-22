# UTKFace dataset loader: HuggingFace-based, 70/10/20 split (train/val/test), age regression labels
import os
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev = (0.229, 0.224, 0.225)
MINIMUM_ACCURACY = 0.01
_cache_dir = os.path.join(str(data_dir), 'utkface')


def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    os.makedirs(_cache_dir, exist_ok=True)
    dataset = load_dataset("nu-delta/utkface", split="train", cache_dir=_cache_dir)
    full_dataset = _UTKFace(dataset, transform)
    n = len(full_dataset)
    train_size = int(n * 0.70)
    val_size   = int(n * 0.10)
    test_size  = n - train_size - val_size  # remaining ~20%
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Attach held-out test set as attribute so Train.py can run a final clean evaluation
    val_ds.held_out_test = test_ds
    return (1,), MINIMUM_ACCURACY, train_ds, val_ds


class _UTKFace(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x = self.transform(img) if self.transform else self._fallback(img)
        return x, torch.tensor([float(item['age'])], dtype=torch.float32)

    @staticmethod
    def _fallback(img):
        import numpy as np
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
