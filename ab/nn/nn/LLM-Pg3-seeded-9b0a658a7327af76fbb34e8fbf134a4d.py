import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class DilatedBlock(nn.Module):
    """Multi-dilation residual block (1,2,4)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return self.act(self.conv3(out) + x)


class FusionBlock(nn.Module):
    """Channel-wise fusion block using adaptive average pooling and addition."""

    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        avg = self.avgpool(x)
        avg = self.conv(avg)
        return x + avg


class Net(nn.Module):
    """Combined architecture using DilatedBlock and FusionBlock for robust denoising."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.in_conv = nn.Conv2d(3, f, 3, padding=1)
        self.b0 = DilatedBlock(f)
        self.fuse1 = FusionBlock(f)
        self.down0 = nn.Conv2d(f, 2 * f, 3, stride=2, padding=1)
        self.b1 = DilatedBlock(2 * f)
        self.fuse2 = FusionBlock(2 * f)
        self.down1 = nn.Conv2d(2 * f, 4 * f, 3, stride=2, padding=1)
        self.b2 = DilatedBlock(4 * f)
        self.fuse3 = FusionBlock(4 * f)
        self.down2 = nn.Conv2d(4 * f, 8 * f, 3, stride=2, padding=1)
        self.bottleneck = DilatedBlock(8 * f)
        self.upsample = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2)
        self.u1 = nn.Conv2d(4 * f, 4 * f, 3, padding=1)
        self.fuse4 = FusionBlock(4 * f)
        self.upsample2 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2)
        self.u2 = nn.Conv2d(2 * f, 2 * f, 3, padding=1)
        self.fuse5 = FusionBlock(2 * f)
        self.upsample3 = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.u3 = nn.Conv2d(f, f, 3, padding=1)
        self.fuse6 = FusionBlock(f)
        self.out_conv = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def forward(self, x):
        identity = x
        h = self.in_conv(x)
        h = self.b0(h)
        h = self.fuse1(h)
        h = self.down0(h)
        h = self.b1(h)
        h = self.fuse2(h)
        h = self.down1(h)
        h = self.b2(h)
        h = self.fuse3(h)
        h = self.down2(h)
        h = self.bottleneck(h)
        h = self.upsample(h)
        h = self.u1(h)
        h = self.fuse4(h)
        h = self.upsample2(h)
        h = self.u2(h)
        h = self.fuse5(h)
        h = self.upsample3(h)
        h = self.u3(h)
        h = self.fuse6(h)
        return torch.clamp(self.out_conv(h) + identity, 0.0, 1.0)

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
