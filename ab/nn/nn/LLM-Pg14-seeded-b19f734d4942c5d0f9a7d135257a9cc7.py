import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr", "batch", "epoch_max", "transform"}

class Net(nn.Module):
    """Hybrid Denoiser combining Multi-Dilation Blocks and LPPooling Blocks."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        channels = 32
        self.in_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.down0 = nn.MaxPool2d(2)
        self.dil_block0 = DilBlock(channels)
        self.lppool0 = LPPoolBlock(channels)
        self.down1 = nn.MaxPool2d(2)
        self.dil_block1 = DilBlock(channels)
        self.lppool1 = LPPoolBlock(channels)
        self.dil_block2 = DilBlock(channels)
        self.up1 = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.out_conv = nn.Conv2d(channels, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(x)
        h = self.down0(h)
        h = self.dil_block0(h)
        h = self.lppool0(h)
        h = self.down1(h)
        h = self.dil_block1(h)
        h = self.lppool1(h)
        h = self.dil_block2(h)
        h = self.up1(h)
        h = self.up2(h)
        return torch.clamp(self.out_conv(h) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 0.0001)
        batch = prm.get("batch", 16)
        epoch_max = prm.get("epoch_max", 50)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch_max, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)


class DilBlock(nn.Module):
    """Multi-dilation channel-preserving residual block (crossover contract)."""

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


class LPPoolBlock(nn.Module):
    """LPPool2d context branch + Hardsigmoid gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.LPPool2d(2, 4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        c = self.ctx(self.pool(h))
        c = nn.functional.interpolate(c, size=h.shape[-2:], mode="nearest")
        return self.conv2(h * self.gate(c)) + x
