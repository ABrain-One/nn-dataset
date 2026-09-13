import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class Refiner(nn.Module):
    """The ONE shared refinement network, applied at every scale. Input is 6 channels: the
    image at this scale plus the upsampled estimate from the coarser scale."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, img, prev):
        return img + self.net(torch.cat([img, prev], dim=1))


class Net(nn.Module):
    """SCALE-RECURRENT denoiser (SRN family, coarse-to-fine with shared weights).

    ONE refiner network runs three times: first on a quarter-resolution image, then at half
    resolution with the upsampled quarter-scale ESTIMATE as a second input, then at full
    resolution with the half-scale estimate. Weight sharing is across SCALES -- distinct from
    DenoiseRecMem, which shares across DEPTH at one resolution, and from the U-Nets, which
    visit scales once with separate weights. The output image itself is the recurrent state.

    Coarse scales see large structure cheaply; the recurrence means every scale benefits from
    the same learned prior, and the parameter count is a third of an unshared pyramid.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.refiner = Refiner(28)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        x4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        e4 = self.refiner(x4, x4)                              # coarsest: prev = itself
        e2 = self.refiner(
            x2, F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False))
        e1 = self.refiner(
            x, F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False))
        return torch.clamp(e1, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + \
                self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            # The refiner receives three gradient contributions per step (one per scale),
            # so clip tighter than the corpus default -- same reasoning as DenoiseRecMem.
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
