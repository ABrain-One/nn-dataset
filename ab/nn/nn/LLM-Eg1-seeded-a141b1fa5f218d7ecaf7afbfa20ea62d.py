import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr", "batch", "epoch_max", "transform"}


class SplitAndMergeBlock(nn.Module):
    """Split channels, process each half separately, merge results back together."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.split = nn.Identity()
        self.conv_a = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_b = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.merge = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x):
        a, b = self.split(x), self.split(x)
        a, b = self.conv_a(a), self.conv_b(b)
        return self.merge(torch.cat((a, b), dim=1))


class EnhancedWeightedL1PSNRLoss(nn.Module):
    def __init__(self, psnr_weight=0.3, freq_weight=0.08, grad_weight=0.15):
        super().__init__()
        self.psnr_weight = psnr_weight
        self.freq_weight = freq_weight
        self.grad_weight = grad_weight

    def _psnr_weight(self, diff):
        err_map = diff.detach().mean(dim=1, keepdim=True)
        weight = 1.0 + self.psnr_weight * err_map / (err_map.mean() + 1e-6)
        return weight

    def _freq_loss(self, pred, target):
        pf = torch.fft.rfft2(pred, norm='ortho').abs()
        tf = torch.fft.rfft2(target, norm='ortho').abs()
        return F.l1_loss(pf, tf)

    def _grad_loss(self, pred, target):
        diff = pred - target
        dx = diff[:, :, :, 1:] - diff[:, :, :, :-1]
        dy = diff[:, :, 1:, :] - diff[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()

    def forward(self, pred, target):
        diff = (pred - target).abs()
        l1 = diff.mean()
        psnr_weight = self._psnr_weight(diff)
        freq_loss = self._freq_loss(pred, target)
        grad_loss = self._grad_loss(pred, target)
        return l1 + self.psnr_weight * (diff * psnr_weight).mean() + self.freq_weight * freq_loss + self.grad_weight * grad_loss


class LiteDenoisingBlock(nn.Module):
    """Lightweight denoising block with an internal channel-reduction bottleneck (f -> f/2 -> f), standard 3x3 convolutions, hardware-native ReLU activations, and a local residual connection."""
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


class Net(nn.Module):
    """Hybrid denoising network combining SplitAndMergeBlock and LiteDenoisingBlock."""
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = f0 * 2, f0 * 4, f0 * 8, f0 * 16

        # Encoder
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = SplitAndMergeBlock(f0, f0)
        self.down0 = nn.Sequential(
            nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.eb1 = SplitAndMergeBlock(f1, f1)
        self.down1 = nn.Sequential(
            nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.eb2 = SplitAndMergeBlock(f2, f2)
        self.down2 = nn.Sequential(
            nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.eb3 = SplitAndMergeBlock(f3, f3)
        self.down3 = nn.Sequential(
            nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.bottleneck = SplitAndMergeBlock(f4, f4)

        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.db3 = LiteDenoisingBlock(f3, f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = LiteDenoisingBlock(f2, f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = LiteDenoisingBlock(f1, f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 3, padding=1)
        self.db0 = LiteDenoisingBlock(f0, f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)

        self.criterion = EnhancedWeightedL1PSNRLoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=prm.get("epoch_max", 50), eta_min=1e-5
        )

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        e3 = self.eb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.db3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d2 = self.db2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), e0], 1)))

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
