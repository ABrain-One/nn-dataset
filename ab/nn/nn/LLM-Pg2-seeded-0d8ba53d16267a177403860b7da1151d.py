import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2): smooth approximation to L1."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

class _DWSResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.act2 = nn.GELU()
        self.conv4 = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.9)

    def forward(self, x):
        net = self.act1(self.dw(self.conv1(x)))
        net = self.conv2(net)
        middle = x * self.alpha + net
        net = self.act2(self.conv3(middle))
        net = self.conv4(net)
        return middle + net



def supported_hyperparameters():
    return {"lr"}


class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


class ProgressiveDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.actv(self.conv(x))


class ProgressiveUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(self.conv(x))


class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dws_res = _DWSResBlock(in_channels)
        self.lite_block = LiteDenoisingBlock(in_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.actv(self.shortcut(x) + self.dws_res(x) + self.lite_block(x))


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.down0 = ProgressiveDownsampling(f0, f1)
        self.down1 = ProgressiveDownsampling(f1, f2)
        self.down2 = ProgressiveDownsampling(f2, f3)
        self.down3 = ProgressiveDownsampling(f3, f4)
        self.bottleneck = HybridBlock(f4, f4)
        self.up3 = ProgressiveUpsampling(f4, f3)
        self.up2 = ProgressiveUpsampling(f3, f2)
        self.up1 = ProgressiveUpsampling(f2, f1)
        self.up0 = ProgressiveUpsampling(f1, f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)
        self.criterion = CharbonnierLoss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e1 = self.down0(e0)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        b = self.bottleneck(self.down3(e3))
        d3 = self.up3(b)
        d2 = self.up2(d3)
        d1 = self.up1(d2)
        d0 = self.up0(d1)
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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
