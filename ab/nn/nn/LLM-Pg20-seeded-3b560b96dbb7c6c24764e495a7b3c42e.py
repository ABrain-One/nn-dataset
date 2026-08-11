import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr"}


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


class MidBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(f, f, 3)
        self.conv2 = nn.Conv2d(f, f, 3)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ca = nn.Sequential(nn.Conv2d(f, f, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(48, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        n = self.ca(self.act(self.conv1(self.pad(x)))) + x
        return self.sa(self.act(self.conv2(self.pad(n)))) + n


class Up(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.up = nn.ConvTranspose2d(f, f, 2, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, y, skip):
        return self.act(self.up(y)) + skip


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.d1(x))
        h = self.act(self.d2(h))
        return self.act(self.d4(h) + x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        f = 48
        self.depth = 4
        self.head = nn.Conv2d(ch, f, 3, padding=1)
        self.downs = nn.ModuleList([SplitDown(f) for _ in range(self.depth)])
        self.mid = MidBlock(f)
        self.body = nn.Sequential(*[ResidualBlock(f) for _ in range(4)], SEBlock(f))
        self.ups = nn.ModuleList([Up(f) for _ in range(self.depth)])
        self.tail = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), SEBlock(f), nn.Conv2d(f, ch, 3, padding=1))
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.head(x)
        skips = []
        for d in self.downs:
            skips.append(h)
            h = d(h)
        h = self.mid(h)
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
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
