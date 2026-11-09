import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        h = self.act(self.conv1(x))
        return self.act(self.conv2(h) + identity)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SplitDown(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.half = f // 2
        self.pad = nn.ReflectionPad2d(1)
        self.conv_a = nn.Conv2d(self.half, f, 3, stride=2)
        self.conv_b = nn.Conv2d(self.half, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        a = x[:, :self.half]
        b = x[:, self.half:]
        a = self.act(self.conv_a(self.pad(a)))
        b = self.act(self.conv_b(self.pad(b)))
        return a + b


class Upsample(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.up = nn.ConvTranspose2d(f, f, 2, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, y, skip):
        return self.act(self.up(y)) + skip


class LitDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ELU(inplace=True)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.ca(self.actv(self.conv2(out)))
        return self.actv(out + identity)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        f = 48
        self.depth = 4
        self.head = nn.Conv2d(ch, f, 3, padding=1)
        self.downs = nn.ModuleList([SplitDown(f) for _ in range(self.depth)])
        self.body = nn.Sequential(*[ResidualBlock(f) for _ in range(4)])
        self.ups = nn.ModuleList([Upsample(f) for _ in range(self.depth)])
        self.tail = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.Conv2d(f, ch, 3, padding=1))
        self.criterion = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.head(x)
        skips = []
        for d in self.downs:
            skips.append(h)
            h = d(h)
        for b in self.body:
            h = b(h)
        for u, s in zip(self.ups, reversed(skips)):
            h = u(h, s)
        return torch.clamp(self.tail(h) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion(preds, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
