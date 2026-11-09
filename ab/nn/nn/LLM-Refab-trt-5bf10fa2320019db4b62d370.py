import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LaplacianPyramidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.low_pass = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1)
        )
        self.high_pass = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        low = self.pool(x)
        high = x - self.up(low)
        low_denoised = self.low_pass(low)
        high_denoised = self.high_pass(high)
        return self.up(low_denoised) + high_denoised


def supported_hyperparameters():
    return {'lr'}

class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.conv1(self.act(self.conv2(x)))
        return h + skip

class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 5, padding=2)
        self.conv2 = nn.Conv2d(mid, out_channels, 1, padding=0)
        self.actv = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return x + self.actv(out + identity)

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.initial_unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, 48, 3, padding=1)
        self.dp_blocks = nn.ModuleList([
            DualPathBlock(48),
            DualPathBlock(48),
            DualPathBlock(48)
        ])
        self.lp_block = LaplacianPyramidBlock(48)
        self.out_conv = nn.Conv2d(48, 3 * 4, 3, padding=1)
        self.output_shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.initial_unshuffle(x)
        h = self.in_conv(h)
        for dp in self.dp_blocks:
            h = dp(h)
        h = self.lp_block(h)
        return torch.clamp(self.output_shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
