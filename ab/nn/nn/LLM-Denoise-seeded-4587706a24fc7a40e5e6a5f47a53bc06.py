import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil


def supported_hyperparameters():
    return {"lr"}


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


class ContextualBlock(nn.Module):
    """Contextual attention block with AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AvgPool2d(4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode="nearest")
        return self.conv2(out + c) + x


class MBConvBlock(nn.Module):
    """Inverted-residual (MobileNetV2-style): expand 1x1 -> depthwise -> project 1x1."""

    def __init__(self, channels):
        super().__init__()
        hidden = channels * 2
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.act(self.expand(x))
        out = self.act(self.dw(out))
        return x + self.project(out)


class Net(nn.Module):
    """Combined architecture using DilatedBlock, ContextualBlock, and MBConvBlock for robust denoising."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        Block = DilatedBlock
        self.in_conv = nn.Conv2d(3, f, 3, padding=1)
        self.b0 = Block(f)
        self.b1 = ContextualBlock(f)
        self.down0 = nn.Conv2d(f, 2 * f, 3, stride=2, padding=1)
        self.b2 = MBConvBlock(2 * f)
        self.down1 = nn.Conv2d(2 * f, 4 * f, 3, stride=2, padding=1)
        self.b3 = Block(4 * f)
        self.down2 = nn.Conv2d(4 * f, 8 * f, 3, stride=2, padding=1)
        self.bottleneck = MBConvBlock(8 * f)
        self.up2 = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2)
        self.b4 = Block(4 * f)
        self.up1 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2)
        self.b5 = MBConvBlock(2 * f)
        self.up0 = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.b6 = Block(f)
        self.out_conv = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(x)
        h = self.b0(h)
        h = self.b1(h)
        h = self.down0(h)
        h = self.b2(h)
        h = self.down1(h)
        h = self.b3(h)
        h = self.down2(h)
        h = self.bottleneck(h)
        h = self.up2(h)
        h = self.b4(h)
        h = self.up1(h)
        h = self.b5(h)
        h = self.up0(h)
        h = self.b6(h)
        return torch.clamp(self.out_conv(h) + identity, 0.0, 1.0)

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
                # BN buffers were updated by this batch's forward BEFORE the
                # check — restore them or eval is poisoned while train stays ok
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