import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReflectPadBlock(nn.Module):
    """ReflectionPad + valid conv + Softplus residual block."""

    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3)
        self.act = nn.Softplus()

    def forward(self, x):
        h = self.act(self.conv1(self.pad(x)))
        return self.conv2(self.pad(h)) + x


def supported_hyperparameters():
    return {'lr'}

class ELUAvgBlock(nn.Module):
    """AdaptiveMaxPool + nn.Linear channel-mixing gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 1, padding=0)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(channels, channels)
        self.act = nn.Softsign()

    def forward(self, x):
        h = self.conv1(x)
        w = self.fc(self.pool(h).flatten(1))
        h = h * self.act(w).unsqueeze(-1).unsqueeze(-1)
        return self.conv2(h) + x

class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.conv1(self.act(self.conv2(x)))
        return h + skip



class RefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.act(self.conv1(self.act(self.conv2(x))))
        return h + skip

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = ELUAvgBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = Block(16)
        self.down0 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.eb1 = DualPathBlock(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.eb2 = ReflectPadBlock(64)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bottleneck = Block(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.db2 = RefineBlock(64)
        self.up1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.db1 = RefineBlock(32)
        self.up0 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.db0 = RefineBlock(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        d2 = self.db2(self.up2(bt))
        d1 = self.db1(self.up1(d2))
        d0 = self.db0(self.up0(d1))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

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
