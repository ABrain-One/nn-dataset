import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, y, skip):
        y = self.deconv(y)
        skip = F.interpolate(skip, y.size()[2:], mode='nearest')
        x = self.act(self.conv(torch.cat([y, skip], 1)))
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv2 = nn.Conv2d(channels // 4, channels // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(channels // 4, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return torch.clamp(x + identity, 0.0, 1.0)

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1] if len(in_shape) >= 3 else 3
        f = 32
        self.depth = 3
        self.head = nn.Conv2d(ch, f, 3, padding=1)
        self.downs = nn.ModuleList([Downsample(f, f) for _ in range(self.depth)])
        self.bottlenecks = nn.ModuleList([Bottleneck(f) for _ in range(self.depth)])
        self.ups = nn.ModuleList([Upsample(f, f) for _ in range(self.depth)])
        self.tail = nn.Conv2d(f, ch, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def forward(self, x):
        identity = x
        h = self.head(x)
        skips = []
        for i, d in enumerate(self.downs):
            skips.append(h)
            h = d(h)
            h = self.bottlenecks[i](h)
        for i, u in enumerate(reversed(self.ups)):
            h = u(h, skips[i])
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
