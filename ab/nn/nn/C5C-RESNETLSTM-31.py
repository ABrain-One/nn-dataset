import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def supported_hyperparameters():
    return {"lr", "momentum"}

# ---------- CBAM primitives ----------
class ChannelAttention(nn.Module):
    """Channel attention: GAP/MaxP -> shared MLP -> sigmoid -> scale per channel."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        mx  = self.max_pool(x).view(b, c)
        att = self.mlp(avg) + self.mlp(mx)
        att = self.sigmoid(att).view(b, c, 1, 1)
        return x * att

class SpatialAttention(nn.Module):
    """Spatial attention: concat(AvgC, MaxC) -> 7x7 conv -> sigmoid -> scale per spatial location."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x

# ---------- Model ----------
def _unpack_chw(shape) -> Tuple[int, int, int]:
    # Accept (C,H,W) or (N,C,H,W) and return (C,H,W)
    if isinstance(shape, (tuple, list)):
        if len(shape) == 3:
            return int(shape[0]), int(shape[1]), int(shape[2])
        if len(shape) >= 4:
            return int(shape[1]), int(shape[2]), int(shape[3])
    raise ValueError(f"Expected in/out shape like (C,H,W) or (N,C,H,W), got {shape}")

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        in_c, in_h, in_w   = _unpack_chw(in_shape)
        out_c, out_h, out_w = _unpack_chw(out_shape)

        # A simple image-to-image head with CBAM in the middle
        mid_c = max(out_c, 32)
        self.body = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            CBAM(mid_c, reduction=int(self.prm.get("reduction_ratio", 16)), spatial_kernel=7),
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=True),
        )

        self.upsample = None
        if (out_h, out_w) != (in_h, in_w):
            self.upsample = nn.Upsample(size=(out_h, out_w), mode="bilinear", align_corners=False)

        # training state
        self.criterion = None
        self.optimizer = None
        self.to(self.device)

    # ---- Training helpers ----
    def train_setup(self, prm):
        prm = prm or {}
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.to(self.device)
        # Image-to-image: use MSE by default.
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        if self.criterion is None or self.optimizer is None:
            self.train_setup(self.prm)

        self.train()
        for inputs, targets in train_data:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
            preds = self.forward(inputs)
            loss = self.criterion(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---- Forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = self.body(x)
        if self.upsample is not None:
            y = self.upsample(y)
        return y
