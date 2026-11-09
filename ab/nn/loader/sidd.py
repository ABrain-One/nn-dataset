import os
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image

# SIDD (Smartphone Image Denoising Dataset), sRGB. Downloaded + converted manually into
# the same layout the MAI denoise loader uses (see data/sidd_to_pairs.py):
#     <SIDD_ROOT>/original/   noisy sRGB images
#     <SIDD_ROOT>/denoised/   clean / ground-truth sRGB images
# paired by sorted filename. Unlike the MAI loader there is NO auto-download: SIDD-Medium
# is a single 320-pair pool, so we deterministically split it into train/val here.
# Overridable so a k8s job can point at the NAS copy without a symlink, exactly as midd.py
# does. Default is unchanged, so any existing caller behaves identically.
SIDD_ROOT = os.environ.get('SIDD_ROOT', os.path.join(os.getcwd(), 'sidd'))
VAL_FRACTION = 0.1          # hold out the last 10% (by sorted name) as the test split
CROP = (256, 256)           # match the MAI denoise recipe so seeds/metrics stay comparable


def _pairs(root):
    noisy_dir = os.path.join(root, 'original')
    clean_dir = os.path.join(root, 'denoised')
    if not (os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)):
        raise FileNotFoundError(
            f"SIDD data not found at {root}. Expected {root}/original and {root}/denoised. "
            f"Download SIDD-Medium sRGB and convert with data/sidd_to_pairs.py first.")
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    noisy = sorted(f for f in os.listdir(noisy_dir) if f.endswith(exts))
    clean = sorted(f for f in os.listdir(clean_dir) if f.endswith(exts))
    if len(noisy) != len(clean):
        raise ValueError(f"SIDD pair count mismatch: {len(noisy)} noisy vs {len(clean)} clean")
    if not noisy:
        raise FileNotFoundError(f"no images under {noisy_dir}")
    return noisy_dir, clean_dir, noisy, clean


class SiddDataset(Dataset):
    def __init__(self, root_dir, indices, transform=None, crop_size=CROP, is_val=False):
        self.noisy_dir, self.clean_dir, noisy, clean = _pairs(root_dir)
        self.noisy_files = [noisy[i] for i in indices]
        self.clean_files = [clean[i] for i in indices]
        self.transform = transform
        self.crop_size = crop_size
        self.is_val = is_val

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert("RGB")
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert("RGB")

        w, h = clean_img.size
        target_h, target_w = self.crop_size

        # Mirror the MAI denoise loader's crop logic so PSNR stays comparable across datasets.
        # Crop 2px larger, then center-crop to target, so pixels keep full 1px neighbours.
        pad_crop_h = target_h + 2
        pad_crop_w = target_w + 2

        if self.is_val:
            i = (h - target_h) // 2
            j = (w - target_w) // 2
            clean_img = F.crop(clean_img, i, j, target_h, target_w)
            noisy_img = F.crop(noisy_img, i, j, target_h, target_w)
        else:
            i = torch.randint(0, h - pad_crop_h + 1, size=(1,)).item()
            j = torch.randint(0, w - pad_crop_w + 1, size=(1,)).item()
            clean_img = F.crop(clean_img, i, j, pad_crop_h, pad_crop_w)
            noisy_img = F.crop(noisy_img, i, j, pad_crop_h, pad_crop_w)
            clean_img = F.center_crop(clean_img, (target_h, target_w))
            noisy_img = F.center_crop(noisy_img, (target_h, target_w))

            if random.random() > 0.5:
                clean_img = F.hflip(clean_img); noisy_img = F.hflip(noisy_img)
            if random.random() > 0.5:
                clean_img = F.vflip(clean_img); noisy_img = F.vflip(noisy_img)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                clean_img = F.rotate(clean_img, angle); noisy_img = F.rotate(noisy_img, angle)

        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, clean_img


def loader(transform_fn, task=None):
    final_transform = transform_fn()

    _, _, noisy, _ = _pairs(SIDD_ROOT)
    n = len(noisy)
    n_val = max(1, int(round(n * VAL_FRACTION)))
    all_idx = list(range(n))
    val_idx = all_idx[-n_val:]          # deterministic: last VAL_FRACTION by sorted name
    train_idx = all_idx[:-n_val]

    train_set = SiddDataset(SIDD_ROOT, train_idx, transform=final_transform, crop_size=CROP, is_val=False)
    test_set = SiddDataset(SIDD_ROOT, val_idx, transform=final_transform, crop_size=CROP, is_val=True)

    return (3, CROP[0], CROP[1]), 0.0, train_set, test_set
