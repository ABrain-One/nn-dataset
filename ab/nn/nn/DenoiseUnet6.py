import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class FreqL1Loss(nn.Module):
    """L1 pixel loss + FFT magnitude loss for high-frequency texture recovery."""
    def __init__(self, freq_weight=0.08):
        super().__init__()
        self.freq_weight = freq_weight

    def forward(self, pred, target):
        pixel = F.l1_loss(pred, target)
        pf = torch.fft.rfft2(pred, norm='ortho').abs()
        tf = torch.fft.rfft2(target, norm='ortho').abs()
        return pixel + self.freq_weight * F.l1_loss(pf, tf)


class _DWSResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.act2 = nn.GELU()
        self.conv4 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        net = self.act1(self.dw(self.conv1(x)))
        net = self.conv2(net)
        middle = x + net
        net = self.act2(self.conv3(middle))
        net = self.conv4(net)
        return middle + net


class _SkipGate(nn.Module):
    """Soft attention gate on skip features before fusion."""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip):
        return skip * self.gate(skip)


class ProgressiveShuffleNet(nn.Module):
    def __init__(self, in_channels=3, base_filters=32):
        super().__init__()
        s = base_filters

        self.init_conv = nn.Conv2d(in_channels, s, 3, 1, 1)

        self.enc1 = _DWSResidualBlock(s)
        self.pool1 = nn.Conv2d(s, s * 2, 1, stride=2)

        self.enc2 = _DWSResidualBlock(s * 2)
        self.pool2 = nn.Conv2d(s * 2, s * 4, 1, stride=2)

        self.enc3 = _DWSResidualBlock(s * 4)
        self.pool3 = nn.Conv2d(s * 4, s * 8, 1, stride=2)

        self.enc4 = _DWSResidualBlock(s * 8)
        self.pool4 = nn.Conv2d(s * 8, s * 16, 1, stride=2)

        self.bottleneck = _DWSResidualBlock(s * 16)

        self.up6 = nn.Conv2d(s * 16, s * 32, 1)
        self.gate6 = _SkipGate(s * 8)
        self.dec6 = _DWSResidualBlock(s * 8)

        self.up7 = nn.Conv2d(s * 8, s * 16, 1)
        self.gate7 = _SkipGate(s * 4)
        self.dec7 = _DWSResidualBlock(s * 4)

        self.up8 = nn.Conv2d(s * 4, s * 8, 1)
        self.gate8 = _SkipGate(s * 2)
        self.dec8 = _DWSResidualBlock(s * 2)

        self.up9 = nn.Conv2d(s * 2, s * 4, 1)
        self.gate9 = _SkipGate(s)
        self.dec9 = _DWSResidualBlock(s)

        self.ps = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(s, 3, 3, 1, 1)

    def _upconv(self, x, skip, up_conv, gate):
        return self.ps(up_conv(x)) + gate(skip)

    def forward(self, x):
        x0 = self.init_conv(x)

        c1 = self.enc1(x0)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))
        c4 = self.enc4(self.pool3(c3))
        c5 = self.bottleneck(self.pool4(c4))

        u6 = self.dec6(self._upconv(c5, c4, self.up6, self.gate6))
        u7 = self.dec7(self._upconv(u6, c3, self.up7, self.gate7))
        u8 = self.dec8(self._upconv(u7, c2, self.up8, self.gate8))
        u9 = self.dec9(self._upconv(u8, c1, self.up9, self.gate9))

        return self.final_conv(u9)


class Net(nn.Module):
    """
    Progressive PixelShuffle U-Net built from depthwise-separable residual
    blocks with attention-gated skip connections at every scale. Trained
    with an L1 + FFT-magnitude (frequency-aware) loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = ProgressiveShuffleNet(in_channels=3, base_filters=32)
        self.criterion = FreqL1Loss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

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
