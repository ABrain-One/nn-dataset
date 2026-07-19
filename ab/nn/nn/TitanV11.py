# Model Architecture: TitanV11
# NPU-Optimized Super-Resolution | Stage 1 Base Training
# BSConvU x8 | Custom DepthToSpace | ReLU Only | 1x1 and 3x3 Kernels Only

import torch
import torch.nn as nn
import torch.nn.functional as F
from ab.nn.util.Classes import DataRoll


def supported_hyperparameters():
    return {"lr"}


# ── Charbonnier Loss ────────────────────────────────────────────
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


# ── BSConvU Block ───────────────────────────────────────────────
# Blueprint Separable Convolution: 1x1 pointwise + 3x3 depthwise + ReLU
class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 standard conv: mixes channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        # 3x3 depthwise conv: spatial feature extraction
        self.depthwise = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1,
            groups=out_channels, bias=False
        )
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        return self.act(x)


# ── Residual BSConv Block ───────────────────────────────────────
# Expand with BSConvU → squeeze with 1x1 → scaled residual add
class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.bsconv   = BSConvU(channels, channels * 2)
        self.squeeze  = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.act      = nn.ReLU()

    def forward(self, x):
        res   = x
        out   = self.bsconv(x)
        out   = self.act(self.squeeze(out))
        return out * 0.1 + res


# ── Custom DepthToSpace ─────────────────────────────────────────
# Replaces nn.PixelShuffle using view + permute
# Input : (B, C * scale^2, H, W)
# Output: (B, C, H*scale, W*scale)
class DepthToSpace(nn.Module):
    def __init__(self, scale=3):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        B, C_s2, H, W = x.shape
        s = self.scale
        C = C_s2 // (s * s)
        # Reshape → permute → merge  (TF spatial packing semantics)
        x = x.view(B, C, s, s, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C, H * s, W * s)
        return x


# ── Main Net Class (TitanV11 Architecture) ──────────────────────
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple,
                 prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.scale  = 3
        channels    = 64

        # Feature extraction head (3x3)
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        # Body: 8 x ResBlock (BSConvU-based)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(8)])

        # Tail: 1x1 projection → DepthToSpace (3x upscale)
        self.exit      = nn.Conv2d(channels, 3 * self.scale * self.scale,
                                   kernel_size=1, bias=False)
        self.upsampler = DepthToSpace(scale=self.scale)

        # Loss
        self.criterion = CharbonnierLoss(eps=1e-3)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out  = self.exit(feat)
        out  = self.upsampler(out)

        # Global Residual Learning: bilinear baseline (NPU hardware-accelerated)
        base = F.interpolate(x, scale_factor=self.scale,
                             mode='bilinear', align_corners=False)
        out  = out + base
        return torch.clamp(out, 0.0, 1.0)

    def train_setup(self, prm):
        learning_rate = prm.get("lr", 2e-4)
        max_epochs = prm.get("epoch_max", 800)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )
        # Kaiming init for all conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.exit.weight, std=0.001)

    def _psnr_rgb(self, sr, hr):
        """Standard RGB PSNR calculation."""
        mse = F.mse_loss(sr, hr)
        return torch.tensor(float('inf')) if mse == 0 \
               else 10.0 * torch.log10(1.0 / mse)

    def learn(self, data_roll: DataRoll):
        self.train()
        total_loss, n = 0.0, 0
        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.forward(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        self.scheduler.step()
        return total_loss / max(n, 1)

    def valid(self, data_roll: DataRoll):
        self.eval()
        total_psnr, n = 0.0, 0
        with torch.no_grad():
            for inputs, labels in data_roll:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                total_psnr += self._psnr_rgb(self.forward(inputs), labels).item()
                n += 1
        avg = total_psnr / max(n, 1)
        print(f"\n--- RGB PSNR: {avg:.4f} dB ---\n")
        return avg
