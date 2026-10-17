import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr'}


class DualLiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.conv3 = nn.Conv2d(mid, out_channels, 5, padding=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        skip = self.shortcut(x)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.conv3(h)
        return F.relu(h + skip)


class LiteDualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.shortcut = nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        return F.relu(h + skip)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 5, padding=2)
        self.ldb0 = LiteDualBlock(f0, f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.01))
        self.ldb1 = LiteDualBlock(f1, f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.01))
        self.ldb2 = LiteDualBlock(f2, f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.01))
        self.ldb3 = LiteDualBlock(f3, f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.01))
        self.bottleneck = DualLiteDenoisingBlock(f4, f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.dld3 = DualLiteDenoisingBlock(f3, f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.dld2 = DualLiteDenoisingBlock(f2, f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.dld1 = DualLiteDenoisingBlock(f1, f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 7, padding=3)
        self.dld0 = DualLiteDenoisingBlock(f0, f0)
        self.out_conv = nn.Conv2d(f0, channels, 5, padding=2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def forward(self, x):
        identity = x
        e0 = self.ldb0(self.in_conv(x))
        e1 = self.ldb1(self.down0(e0))
        e2 = self.ldb2(self.down1(e1))
        e3 = self.ldb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dld3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d2 = self.dld2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d1 = self.dld1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.dld0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
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
