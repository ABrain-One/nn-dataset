import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr"}


class ResidualConvBlock(nn.Module):
    """Channel-preserving residual block with ELU activation."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return torch.add(identity, out)


class UpsampleBlock(nn.Module):
    """Pixel shuffle based upsampling block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class DownsampleBlock(nn.Module):
    """Downsampling block using strided convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    """Combine ResidualConvBlock and UpsampleBlock to form a small, efficient denoiser."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.in_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.down1 = DownsampleBlock(32, 64)
        self.res1 = ResidualConvBlock(64)
        self.up1 = UpsampleBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.up1(x)
        return torch.clamp(self.out_conv(x) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean)
            bad = not torch.isfinite(loss)
            if bad:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)
