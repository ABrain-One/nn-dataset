"""MAI denoise images with SYNTHETIC AWGN at a controllable level.

WHY A SECOND BENCHMARK. The MAI pairs are nearly clean: returning the noisy input unchanged
scores 33.02 dB, the best of 1,462 searched models scores 35.15, so the entire achievable
range is about 4 dB. Measured consequences, all pointing the same way:

    73% of the corpus scores between 33 and 34 dB
    minimum pairwise error-correlation across 317 models is 0.693 -- nothing behaves unlike
      anything else
    a do-nothing identity baseline correlates 0.930 with trained models, ABOVE the 0.923 they
      manage with each other

Architectures cannot distinguish themselves on a task with no room in it. The generators are
not failing; they are being scored through a metric that cannot resolve them.

A noise sweep on these same images confirmed the models learned real denoising and are simply
being under-measured: gain over no-op rises monotonically -0.09 / +1.42 / +1.78 / +1.97 /
+2.00 dB at sigma 5 / 10 / 15 / 25 / 50. At sigma 25 the headroom is roughly 9 dB instead of 4.

SAME IMAGES, ONLY THE NOISE CHANGES. Built from MAI's own `denoised/` directories -- which are
the clean ground truth -- rather than DIV2K/CBSD68, for two reasons. It needs no data we do
not already have (ab/nn/loader/awgn_color.py wants a data/awgn tree that does not exist here),
and holding the content fixed makes noise level the ONLY variable. A benchmark that changed
content and difficulty at once could not tell you which one moved the result.

VALIDATION NOISE IS DETERMINISTIC, and this is the part that would silently ruin everything.
If each epoch drew fresh noise, every model would be scored on a different test and the
differences between two 34 dB models would be sampling noise, not architecture. Each
validation image gets noise seeded from its own index, so the test set is FIXED: byte-identical
across models, epochs, runs and machines. Training noise stays random -- there it is
augmentation and resampling is the point.

    AWGN_SIGMA   noise std on the 0-255 scale (default 25)

Registered as its own dataset name, so its rows live beside the originals in `stat` rather
than overwriting them. Scores from the two benchmarks are NOT comparable and the separate
dataset key is what stops them being mixed by accident.
"""
import os

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT_TRAIN = os.path.join(os.getcwd(), 'train')
DATASET_ROOT_VAL = os.path.join(os.getcwd(), 'validation')
SIGMA = float(os.environ.get("AWGN_SIGMA", "25"))
CROP = (256, 256)
_EXT = ('.png', '.jpg', '.jpeg')


class AWGNFromClean(Dataset):
    """Clean MAI images, noise added at load time."""

    def __init__(self, root_dir, transform=None, crop_size=CROP, is_val=False,
                 sigma=SIGMA):
        # The CLEAN directory is used for both sides: it is the target, and the source the
        # noise is added to. `original/` (MAI's own noisy capture) is deliberately unused --
        # mixing real sensor noise with synthetic Gaussian would make the noise level
        # uncontrolled, which is the one thing this loader exists to control.
        self.clean_path = os.path.join(root_dir, 'denoised')
        self.files = sorted(f for f in os.listdir(self.clean_path)
                            if f.lower().endswith(_EXT))
        if not self.files:
            raise FileNotFoundError(f"no clean images under {self.clean_path}")
        self.transform = transform
        self.crop_size = crop_size
        self.is_val = is_val
        self.sigma = sigma / 255.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clean = Image.open(os.path.join(self.clean_path, self.files[idx])).convert("RGB")
        w, h = clean.size
        th, tw = self.crop_size

        if self.is_val:
            # Centre crop: the exact region every stored MAI result was measured on, so the
            # two benchmarks differ in noise and nothing else.
            i, j = (h - th) // 2, (w - tw) // 2
            clean = F.crop(clean, i, j, th, tw)
        else:
            i = torch.randint(0, max(1, h - th + 1), size=(1,)).item()
            j = torch.randint(0, max(1, w - tw + 1), size=(1,)).item()
            clean = F.crop(clean, i, j, th, tw)
            if torch.rand(1).item() > 0.5:
                clean = F.hflip(clean)
            if torch.rand(1).item() > 0.5:
                clean = F.vflip(clean)

        clean_t = self.transform(clean) if self.transform else F.to_tensor(clean)

        if self.is_val:
            # Per-image seed -> the same noise every time this index is read, on any machine.
            g = torch.Generator().manual_seed(1000003 + idx)
            noise = torch.randn(clean_t.shape, generator=g) * self.sigma
        else:
            noise = torch.randn_like(clean_t) * self.sigma

        noisy_t = (clean_t + noise).clamp(0.0, 1.0)
        return noisy_t, clean_t


def loader(transform_fn, task=None):
    """LEMUR contract: -> ((C,H,W), minimum_accuracy, train_set, test_set)."""
    tf = transform_fn() if callable(transform_fn) else transform_fn
    sigma = float(os.environ.get("AWGN_SIGMA", SIGMA))
    print(f"[denoise_awgn] sigma={sigma:.0f}/255 on MAI clean images "
          f"| val noise is seeded per image (fixed test set)", flush=True)
    train_set = AWGNFromClean(DATASET_ROOT_TRAIN, transform=tf, is_val=False, sigma=sigma)
    test_set = AWGNFromClean(DATASET_ROOT_VAL, transform=tf, is_val=True, sigma=sigma)
    return (3, CROP[0], CROP[1]), 0.0, train_set, test_set
