import torch
import torch.nn as nn
import torch.nn.functional as F
from ab.nn.util.Classes import DataRoll

def supported_hyperparameters():
    return {"lr"}


# ─────────────────────────────────────────────────────────────
# Charbonnier Loss
# ─────────────────────────────────────────────────────────────
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


# ─────────────────────────────────────────────────────────────
# BSConvU: Blueprint Separable Convolution (NPU-safe)
# 1x1 pointwise → 3x3 depthwise → ReLU
# ─────────────────────────────────────────────────────────────
class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Step 1: 1x1 pointwise conv to mix channels
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Step 2: 3x3 depthwise conv (groups = out_channels)
        self.dw = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            padding=1, groups=out_channels, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pw(x)
        x = self.dw(x)
        return self.act(x)


# ─────────────────────────────────────────────────────────────
# EdgeConvBlock: Residual block built with BSConvU
# ─────────────────────────────────────────────────────────────
class EdgeConvBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # Expand → Depthwise → Squeeze using BSConvU blocks
        self.conv1 = BSConvU(channels, channels * 2)
        # 1x1 conv to squeeze back down (NPU-safe pointwise)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.act   = nn.ReLU()

    def forward(self, x):
        res   = x
        x_out = self.conv1(x)
        x_out = self.act(self.conv2(x_out))
        # Scaled residual connection (0.1 scale = stable training)
        return x_out * 0.1 + res


# ─────────────────────────────────────────────────────────────
# Custom DepthToSpace (replaces nn.PixelShuffle)
# Uses reshape + permute → fully NPU-traceable on Hexagon
# scale_factor = 3  →  output = (B, C, H*3, W*3)
# ─────────────────────────────────────────────────────────────
class DepthToSpace(nn.Module):
    def __init__(self, scale: int = 3):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        # x shape: (B, C * scale^2, H, W)
        B, C_s2, H, W = x.shape
        s  = self.scale
        C  = C_s2 // (s * s)          # output channels (e.g., 3)

        # Reshape → (B, C, s, s, H, W)
        x = x.view(B, C, s, s, H, W)
        # Permute → (B, C, H, s, W, s)  — TF-style spatial packing
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        # Merge spatial dims → (B, C, H*s, W*s)
        x = x.view(B, C, H * s, W * s)
        return x


# ─────────────────────────────────────────────────────────────
# TitanV10 Architecture (NPU-Optimized)
# ─────────────────────────────────────────────────────────────
class TitanV10(nn.Module):
    def __init__(self, channels=64, scale=3):
        super().__init__()
        self.scale = scale

        # Head: standard 3x3 conv for feature extraction
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        # Body: 16 BSConvU-based EdgeConv residual blocks
        self.body = nn.Sequential(*[EdgeConvBlock(channels) for _ in range(16)])

        # Tail: 1x1 conv to produce scale^2 * out_channels feature maps
        self.exit = nn.Conv2d(channels, 3 * scale * scale, kernel_size=1, bias=False)

        # Custom DepthToSpace upsampler (replaces PixelShuffle)
        self.upsampler = DepthToSpace(scale=scale)

    def forward(self, x):
        features = self.head(x)
        features = self.body(features)

        out  = self.exit(features)
        out  = self.upsampler(out)

        # Global Residual Learning: bilinear baseline (NPU-hardware-accelerated)
        base = F.interpolate(x, scale_factor=self.scale,
                             mode='bilinear', align_corners=False)
        out  = out + base

        # Clamp to valid pixel range [0, 1] for normalized input
        return torch.clamp(out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# Net Wrapper (Training / Validation Manager)
# ─────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple,
                 prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.scale  = 3

        self.model         = TitanV10(channels=64, scale=self.scale).to(device)
        self.criterion_soft = CharbonnierLoss(eps=1e-3)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=400, eta_min=1e-6
        )
        # Weight initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.model.exit.weight, std=0.001)

    # ── Y-Channel PSNR Metric ──────────────────────────────
    def calculate_psnr_y(self, sr, hr):
        """PSNR on Y-channel (Luminance) — matches competition metric."""
        def rgb_to_y(img):
            R, G, B = img[:, 0], img[:, 1], img[:, 2]
            return 0.257 * R + 0.504 * G + 0.098 * B + (16.0 / 255.0)

        sr_y = rgb_to_y(sr)
        hr_y = rgb_to_y(hr)
        mse  = F.mse_loss(sr_y, hr_y)
        if mse == 0:
            return torch.tensor(float('inf')).to(self.device)
        return 10.0 * torch.log10(1.0 / mse)

    # ── Training Loop ──────────────────────────────────────
    def learn(self, data_roll: DataRoll):
        self.model.train()
        total_loss, num_batches = 0.0, 0

        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.criterion_soft(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / max(num_batches, 1)

    # ── Validation Loop ────────────────────────────────────
    def valid(self, data_roll: DataRoll):
        self.model.eval()
        total_psnr, num_batches = 0.0, 0

        with torch.no_grad():
            for inputs, labels in data_roll:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs    = self.model(inputs)
                psnr_y     = self.calculate_psnr_y(outputs, labels)
                total_psnr += psnr_y.item()
                num_batches += 1

        avg_psnr = total_psnr / max(num_batches, 1)
        print(f"\n--- Validation Y-Channel PSNR: {avg_psnr:.4f} dB ---\n")
        return avg_psnr
