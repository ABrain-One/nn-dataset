import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteDenoisingBlock(nn.Module):
    """
    Lightweight denoising block with an internal channel-reduction
    bottleneck (f -> f/2 -> f), standard 3x3 convolutions, hardware-
    native ReLU activations, and a local residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 1, padding=0)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


def supported_hyperparameters():
    return {'lr'}

class LiteSkipConvBlock(nn.Module):
    """Lightweight skip-connection block with depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.bn(self.pointwise(F.relu(self.depthwise(x))))
        return torch.clamp(out + identity, 0.0, 1.0)

class Net(nn.Module):
    """Combined architecture using LiteDenoisingBlock and LiteSkipConvBlock."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = LiteDenoisingBlock(f0, f0)
        self.sk0 = LiteSkipConvBlock(f0, f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = LiteDenoisingBlock(f1, f1)
        self.sk1 = LiteSkipConvBlock(f1, f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = LiteDenoisingBlock(f2, f2)
        self.sk2 = LiteSkipConvBlock(f2, f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb3 = LiteDenoisingBlock(f3, f3)
        self.sk3 = LiteSkipConvBlock(f3, f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = LiteDenoisingBlock(f4, f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.db3 = LiteDenoisingBlock(f3, f3)
        self.sk3_up = LiteSkipConvBlock(f3, f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = LiteDenoisingBlock(f2, f2)
        self.sk2_up = LiteSkipConvBlock(f2, f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = LiteDenoisingBlock(f1, f1)
        self.sk1_up = LiteSkipConvBlock(f1, f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 3, padding=1)
        self.db0 = LiteDenoisingBlock(f0, f0)
        self.sk0_up = LiteSkipConvBlock(f0, f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)
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
        e0 = self.eb0(self.in_conv(x))
        s0 = self.sk0(e0)
        e1 = self.eb1(self.down0(e0))
        s1 = self.sk1(e1)
        e2 = self.eb2(self.down1(e1))
        s2 = self.sk2(e2)
        e3 = self.eb3(self.down2(e2))
        s3 = self.sk3(e3)
        b = self.bottleneck(self.down3(e3))
        d3 = self.db3(self.up3(torch.cat([self.upsample(b), s3], 1)))
        s3_up = self.sk3_up(d3)
        d2 = self.db2(self.up2(torch.cat([self.upsample(d3), s2], 1)))
        s2_up = self.sk2_up(d2)
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), s1], 1)))
        s1_up = self.sk1_up(d1)
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), s0], 1)))
        s0_up = self.sk0_up(d0)
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
