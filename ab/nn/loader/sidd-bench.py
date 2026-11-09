"""SIDD sRGB validation on the OFFICIAL protocol -- the one published numbers use.

WHY THIS EXISTS ALONGSIDE sidd.py. `sidd.py` evaluates on a split we invented: the last 10% of
SIDD-Medium by filename (`VAL_FRACTION = 0.1`), randomly cropped to 256x256. That is fine for
comparing our own models against each other, and it is what the cross-dataset study used. It is
NOT comparable to any published SIDD result, because no published result uses it. Every SIDD
number in the denoising literature comes from one of two fixed block sets distributed by the
dataset authors:

    ValidationNoisyBlocksSrgb.mat / ValidationGtBlocksSrgb.mat   ground truth available
    BenchmarkNoisyBlocksSrgb.mat                                 ground truth withheld, submit online

This loader uses the Validation pair. Shape, verified on the downloaded files:

    (40, 32, 256, 256, 3) uint8   = 40 scene instances x 32 blocks = 1280 blocks

Nothing is cropped, resized, shuffled or augmented. Block (i, j) is always block (i, j), so two
runs of two different models see byte-identical inputs and the numbers are directly comparable
to the literature.

TRAIN/TEST CONTAMINATION -- WHY THERE IS NONE, AND THE LIMIT OF THAT CLAIM. SIDD has 200 scene
instances. SIDD-Medium, which is what our training crops come from (the source paths read
`SIDD_Medium_Srgb/Data/0001_001_S6_.../0001_NOISY_SRGB_010.PNG`), contains 160 of them; the
Validation and Benchmark blocks are drawn from the other 40. The two sets are disjoint BY
CONSTRUCTION of the distribution, not by anything this file checks -- the .mat carries pixel
arrays only, with no scene identifiers, so the disjointness cannot be verified from the file
itself. Our training pool spans 158 distinct scene ids, consistent with SIDD-Medium. If training
data is ever added from SIDD-Full or from the validation scenes, this guarantee breaks silently
and every number here becomes worthless. Anyone extending the training set must re-check that.

TRAINING IS NOT SUPPORTED HERE. `loader` returns the same 1280 blocks as both the train and the
test set, which is deliberate: this loader is for EVALUATING an already-trained checkpoint on the
published protocol, never for fitting. Training against it would be training on the test set. The
train slot cannot simply be left empty because the harness expects a dataset object, so it is
filled and guarded -- SIDD_BENCH_ALLOW_TRAIN must be set to use it, and the guard exists so the
mistake has to be deliberate rather than accidental.

Point it at the .mat directory with SIDD_BENCH_ROOT (default: the repo-adjacent data_sidd_bench).
Download them from  http://130.63.97.225/share/SIDD_Blocks/  (~230 MB each).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

BLOCK = (256, 256)
DEFAULT_ROOT = os.environ.get(
    "SIDD_BENCH_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "data_sidd_bench"))
NOISY_MAT = "ValidationNoisyBlocksSrgb.mat"
GT_MAT = "ValidationGtBlocksSrgb.mat"
NOISY_KEY = "ValidationNoisyBlocksSrgb"
GT_KEY = "ValidationGtBlocksSrgb"


def _load_blocks(root):
    """-> (noisy, clean) as uint8 arrays of shape (N, 256, 256, 3), N = 1280."""
    import scipy.io as sio

    root = os.path.abspath(root)
    np_path, gt_path = os.path.join(root, NOISY_MAT), os.path.join(root, GT_MAT)
    for p in (np_path, gt_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"SIDD benchmark blocks not found: {p}\n"
                f"Download from http://130.63.97.225/share/SIDD_Blocks/ "
                f"or set SIDD_BENCH_ROOT to the directory holding them.")
    noisy = sio.loadmat(np_path)[NOISY_KEY]
    clean = sio.loadmat(gt_path)[GT_KEY]
    if noisy.shape != clean.shape:
        raise ValueError(f"block shape mismatch {noisy.shape} vs {clean.shape}")
    if noisy.ndim != 5 or noisy.shape[2:] != (BLOCK[0], BLOCK[1], 3):
        raise ValueError(f"unexpected SIDD block shape {noisy.shape}; "
                         f"expected (scenes, blocks, 256, 256, 3)")
    n = noisy.shape[0] * noisy.shape[1]
    # Flatten scene-major so index order is deterministic and reproducible.
    return (noisy.reshape(n, *noisy.shape[2:]),
            clean.reshape(n, *clean.shape[2:]))


class SiddBenchBlocks(Dataset):
    """The 1280 official blocks, in fixed order. No crop, no flip, no shuffle."""

    def __init__(self, root=None, transform=None):
        self.noisy, self.clean = _load_blocks(root or DEFAULT_ROOT)
        self.transform = transform

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        # uint8 HWC -> float CHW in [0,1]; np.ascontiguousarray because the reshape above can
        # hand back a non-contiguous view and from_numpy would then copy unpredictably.
        n = torch.from_numpy(np.ascontiguousarray(self.noisy[idx])).permute(2, 0, 1).float() / 255.0
        c = torch.from_numpy(np.ascontiguousarray(self.clean[idx])).permute(2, 0, 1).float() / 255.0
        if self.transform is not None:
            # A transform that crops or flips would break protocol comparability, so it is applied
            # to BOTH tensors identically or not at all. Normalisation-only transforms are safe.
            n, c = self.transform(n), self.transform(c)
        return n, c


def loader(transform_fn, task=None):
    final_transform = transform_fn()
    ds = SiddBenchBlocks(transform=final_transform)

    allow = os.environ.get("SIDD_BENCH_ALLOW_TRAIN") == "1"
    if not allow:
        class _NoTrain(Dataset):
            """Refuses to be iterated. Fitting on the published validation blocks would be
            training on the test set, and it should fail loudly rather than quietly inflate a
            number that gets compared to the literature."""

            def __len__(self):
                return len(ds)

            def __getitem__(self, idx):
                raise RuntimeError(
                    "sidd_bench is an EVALUATION-ONLY loader: these are the official SIDD "
                    "validation blocks that published PSNR is measured on. Training on them is "
                    "training on the test set. Use loader 'sidd' to train, then evaluate here. "
                    "Set SIDD_BENCH_ALLOW_TRAIN=1 only if you genuinely intend this.")
        train_set = _NoTrain()
    else:
        train_set = ds

    return (3, BLOCK[0], BLOCK[1]), 0.0, train_set, ds
