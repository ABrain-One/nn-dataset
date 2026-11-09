import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class DWSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.dw = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)
        self.act = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        return self.conv2(self.act(self.dw(self.conv1(x))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.in_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.down1 = DownsampleBlock(32, 64)
        self.res1 = DWSConvBlock(64, 64)
        self.up1 = UpsampleBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.criterion = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def forward(self, x):
        identity = x
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.up1(x)
        return torch.clamp(self.out_conv(x) + identity, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion(preds, clean)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
