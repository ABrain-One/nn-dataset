"""Color AWGN denoising benchmark loader (head-to-head vs HiNAS / DnCNN / FFDNet).

Synthesizes additive white Gaussian noise from CLEAN RGB images at load time
(the standard DnCNN protocol) — noisy = clean + N(0, sigma/255).

  train : DIV2K (+ BSD400) clean HR crops
  test  : CBSD68 clean (Kodak24 / Urban100 evaluated separately at full-train)

Noise level sigma (0-255 scale) is read from env AWGN_SIGMA (default 25); use a
distinct value per benchmark point (15 / 25 / 50) so the framework caches each
separately. Contract matches ab/nn/loader/denoise.py:
    loader(transform_fn, task=None) -> ((C,H,W), 0.0, train_set, test_set)
    dataset __getitem__ -> (noisy, clean)
"""
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

_DATA = os.environ.get("AWGN_DATA_ROOT",
                       "/a/mm/nn-dataset/data/awgn")
TRAIN_CLEAN = os.path.join(_DATA, os.environ.get("AWGN_TRAIN", "train_clean"))  # clean sources
TEST_CLEAN = os.path.join(_DATA, os.environ.get("AWGN_TEST", "cbsd68"))         # clean test set
SIGMA = float(os.environ.get("AWGN_SIGMA", "25")) / 255.0
_EXT = (".png", ".jpg", ".jpeg", ".bmp")


class AWGNDataset(Dataset):
    def __init__(self, root, transform=None, crop=(256, 256), is_val=False, gray=False):
        self.files = sorted(os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(_EXT)) if os.path.isdir(root) else []
        self.transform, self.crop, self.is_val, self.gray = transform, crop, is_val, gray
        # Patch-based training: sample AWGN_CROPS random crops per image per epoch, so an
        # epoch has enough iterations to converge (denoising needs thousands of patches,
        # not one 256-crop per image). Val stays one deterministic center-crop per image.
        self.mult = 1 if is_val else max(1, int(os.environ.get("AWGN_CROPS", "1")))

    def __len__(self):
        return len(self.files) * self.mult

    def __getitem__(self, idx):
        idx = idx % len(self.files)
        mode = "L" if self.gray else "RGB"
        clean = Image.open(self.files[idx]).convert(mode)
        th, tw = self.crop
        w, h = clean.size
        if not self.is_val:
            # random crop + flips/rot for training (DnCNN-style augmentation)
            if w < tw or h < th:
                clean = TF.resize(clean, (max(th, h), max(tw, w)))
                w, h = clean.size
            i, j = random.randint(0, h - th), random.randint(0, w - tw)
            clean = TF.crop(clean, i, j, th, tw)
            if random.random() < 0.5: clean = TF.hflip(clean)
            if random.random() < 0.5: clean = TF.vflip(clean)
        else:
            clean = TF.center_crop(clean, (min(th, h), min(tw, w)))
        clean = self.transform(clean) if self.transform else TF.to_tensor(clean)
        # deterministic noise per test image for a stable metric; random for train
        if self.is_val:
            g = torch.Generator().manual_seed(idx)
            noise = torch.randn(clean.shape, generator=g) * SIGMA
        else:
            noise = torch.randn_like(clean) * SIGMA
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean


def loader(transform_fn, task=None):
    final_transform = transform_fn()
    cs = int(os.environ.get("AWGN_CROP", "256"))   # 128 = faster patch-based AWGN training
    crop = (cs, cs)
    ch = 1 if os.environ.get("AWGN_GRAY") == "1" else 3
    gray = ch == 1
    train_set = AWGNDataset(TRAIN_CLEAN, final_transform, crop, is_val=False, gray=gray)
    test_set = AWGNDataset(TEST_CLEAN, final_transform, crop, is_val=True, gray=gray)
    return (ch, crop[0], crop[1]), 0.0, train_set, test_set
