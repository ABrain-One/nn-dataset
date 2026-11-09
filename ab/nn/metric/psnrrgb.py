import torch
import math

class Net:
    """RGB PSNR metric matching the SplitterNet paper (Flepp/Ignatov CVPR2024, utils.py):
      per-image PSNR = 20*log10(255/sqrt(MSE)) over ALL RGB channels (no Y-channel, no border
      shave), averaged over images. Reported on the 0-1 scale (10*log10(1/mse), max=1), then
      normalized by 48 dB to fit the framework's accuracy convention (real dB = accuracy*48).

    Use this (metric='psnrrgb') for a like-for-like comparison against their 37.92 dB; the
    default 'psnr' is Y-channel and runs ~0.3-0.5 dB higher, so is NOT directly comparable.
    """
    def __init__(self, out_shape=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(self):
        self.total_psnr = 0.0
        self.count = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.clamp(0, 1).to(self.device)
        labels = labels.to(self.device)
        # per-image MSE over C,H,W (all RGB channels)
        mse = torch.mean((outputs - labels) ** 2, dim=[1, 2, 3])   # [B]
        for m in mse:
            v = m.item()
            psnr = 100.0 if v == 0 else 10.0 * math.log10(1.0 / v)
            self.total_psnr += psnr
            self.count += 1

    def compute(self):
        if self.count == 0:
            return 0.0
        avg_psnr = self.total_psnr / self.count       # real dB
        return min(avg_psnr / 48.0, 1.0)              # normalized (real dB = accuracy*48)

    def __call__(self, outputs, labels):
        self.update(outputs, labels)
        return self.compute()

    def result(self):
        return self.compute()


def create_metric(out_shape):
    return Net(out_shape)


Metric = Net
