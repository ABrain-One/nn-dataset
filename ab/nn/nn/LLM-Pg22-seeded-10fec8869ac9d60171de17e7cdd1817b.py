import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class DualPathSPBlock(nn.Module):
    """Dual-path channel-preserving residual block."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.c2 = nn.Conv2d(channels // 2, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.skip_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        identity = self.skip_proj(x)
        h = self.act(self.c1(x))
        h = self.act(self.c2(h))
        return torch.clamp(identity + h, 0.0, 1.0)


class LiteDenoisingBlock(nn.Module):
    """
    Lightweight denoising block with an internal channel-reduction
    bottleneck (f -> f/2 -> f), standard 3x3 convolutions, hardware-
    native ReLU activations, and a local residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 5, padding=2)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return x + self.actv(out + identity)


class Net(nn.Module):
    """
    Combines Dual-Path Sub-Pixel denoising and Lightweight denoising block architectures.
    Upsampled features are concatenated with corresponding levels before feeding back through
    the Dual-Path blocks. Uses PixelUnshuffle for downsampling.
    """

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 48
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 3, padding=1)
        
        self.dpb0 = DualPathSPBlock(f)
        self.lit0 = LiteDenoisingBlock(f, f)
        self.dpb1 = DualPathSPBlock(f)
        self.lit1 = LiteDenoisingBlock(f, f)
        self.dpb2 = DualPathSPBlock(f)
        self.lit2 = LiteDenoisingBlock(f, f)
        self.dpb3 = DualPathSPBlock(f)
        self.lit3 = LiteDenoisingBlock(f, f)
        self.dpb4 = DualPathSPBlock(f)
        self.lit4 = LiteDenoisingBlock(f, f)
        self.dpb5 = DualPathSPBlock(f)
        self.lit5 = LiteDenoisingBlock(f, f)

        self.out_conv = nn.Conv2d(f, 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h = self.dpb0(h)
        h = self.lit0(h)
        h = self.dpb1(h)
        h = self.lit1(h)
        h = self.dpb2(h)
        h = self.lit2(h)
        h = self.dpb3(h)
        h = self.lit3(h)
        h = self.dpb4(h)
        h = self.lit4(h)
        h = self.dpb5(h)
        h = self.lit5(h)
        return torch.clamp(self.shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_mse(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
