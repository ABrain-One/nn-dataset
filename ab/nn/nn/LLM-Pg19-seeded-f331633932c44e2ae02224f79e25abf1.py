import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr"}


class HaarDown(nn.Module):
    def __init__(self, c):
        super().__init__()
        base = torch.tensor([[[1., 1.], [1., 1.]],
                             [[1., -1.], [1., -1.]],
                             [[1., 1.], [-1., -1.]],
                             [[1., -1.], [-1., 1.]]], dtype=torch.float32) / 2.0
        self.register_buffer("w", base.unsqueeze(1).repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        return F.conv2d(x, self.w, stride=2, groups=self.c)


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""

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
    """Squeeze-and-Excitation block."""

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
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.h1 = HaarDown(ch)
        self.h2 = HaarDown(4 * ch)
        f = 16 * ch
        self.enc = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1),
            ResidualBlock(f),
            SEBlock(f)
        )
        self.body = nn.Sequential(
            ResidualBlock(f),
            ResidualBlock(f),
            ResidualBlock(f),
            ResidualBlock(f),
            SEBlock(f)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1),
            SEBlock(f)
        )
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.s1 = nn.Conv2d(ch, 16, 3, padding=1)
        self.s2 = nn.Conv2d(16, 16, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.r1 = nn.Conv2d(16 + ch, 32, 3, padding=1)
        self.r2 = nn.Conv2d(32, ch, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        t = self.enc(self.h2(self.h1(x)))
        t = self.body(t)
        t = self.dec(self.body(t))
        t = self.up2(self.up1(t))
        s = self.act(self.s2(self.act(self.s1(x))))
        r = self.act(self.r1(torch.cat([s, t], dim=1)))
        return torch.clamp(self.r2(r) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
