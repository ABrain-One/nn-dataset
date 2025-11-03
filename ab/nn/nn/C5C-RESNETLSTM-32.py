import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def supported_hyperparameters():
    return {"lr", "momentum"}

# ---------- helpers ----------
def _unpack_chw(shape) -> Tuple[int, int, int]:
    # Accept (C,H,W) or (N,C,H,W) and return (C,H,W)
    if isinstance(shape, (tuple, list)):
        if len(shape) == 3:
            return int(shape[0]), int(shape[1]), int(shape[2])
        if len(shape) >= 4:
            return int(shape[1]), int(shape[2]), int(shape[3])
    raise ValueError(f"Expected (C,H,W) or (N,C,H,W), got {shape}")

# ---------- patchify / unpatchify ----------
class PatchEmbedding(nn.Module):
    """
    Conv2d with kernel=stride=patch_size -> tokens of dim=hidden_dim.
    Returns tokens [B, N, hidden_dim] and the token grid size (H', W').
    """
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W], H/W must be divisible by patch_size
        B, C, H, W = x.shape
        ps = self.patch_size
        if (H % ps) != 0 or (W % ps) != 0:
            raise AssertionError(
                f"H/W must be divisible by patch_size={ps} (got H={H}, W={W})."
            )
        x = self.proj(x)                   # [B, D, H', W']
        B, D, Hp, Wp = x.shape
        x = x.reshape(B, D, Hp * Wp).permute(0, 2, 1)  # [B, N, D]
        return x, (Hp, Wp)

def _unpatchify(tokens: torch.Tensor, grid_hw: Tuple[int, int], out_channels: int, patch_size: int) -> torch.Tensor:
    """
    tokens: [B, N, out_channels * (ps*ps)]
    grid_hw: (Hp, Wp)
    return: [B, out_channels, Hp*ps, Wp*ps]
    """
    B, N, Z = tokens.shape
    Hp, Wp = grid_hw
    ps = patch_size
    expect = out_channels * ps * ps
    assert Z == expect, f"Per-token dim must be out_channels*ps*ps={expect}, got {Z}"
    assert N == Hp * Wp, f"N must equal Hp*Wp ({Hp}*{Wp}) but got {N}"

    x = tokens.view(B, Hp, Wp, out_channels, ps, ps)        # [B, Hp, Wp, C, ps, ps]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()            # [B, C, Hp, ps, Wp, ps]
    x = x.view(B, out_channels, Hp * ps, Wp * ps)           # [B, C, H, W]
    return x

# ---------- Net ----------
class Net(nn.Module):
    """
    Minimal ViT-style encoder:
      - If out_shape is 1D (C,), acts as classifier -> logits [B, C].
      - If out_shape is 3D (C,H,W), acts as image-to-image via token->patch->image unpatchify.
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        in_c, in_h, in_w = _unpack_chw(in_shape)

        # Determine head type from out_shape
        if isinstance(out_shape, (tuple, list)):
            if len(out_shape) == 1:        # classification logits
                self.head_type = "class"
                self.out_dim = int(out_shape[0])
                out_c = self.out_dim
                out_h = out_w = None
            elif len(out_shape) == 3:      # image-to-image
                self.head_type = "image"
                out_c, out_h, out_w = map(int, out_shape)
                self.out_dim = out_c
            else:
                raise ValueError(f"Unsupported out_shape {out_shape}")
        else:
            # scalar/int => classification
            self.head_type = "class"
            self.out_dim = int(out_shape)
            out_c = self.out_dim
            out_h = out_w = None

        # Hyperparams
        patch_size  = int(self.prm.get("patch_size", 16))
        hidden_dim  = int(self.prm.get("hidden_dim", 256))
        dropout_p   = float(self.prm.get("dropout", 0.0))

        # Encoder: patchify -> tokens [B, N, D]
        self.patch = PatchEmbedding(in_channels=in_c, hidden_dim=hidden_dim, patch_size=patch_size)

        # Lightweight token mixer (2-layer MLP over tokens)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        if self.head_type == "class":
            self.pool = nn.AdaptiveAvgPool1d(1)           # over token dimension
            self.cls_head = nn.Linear(hidden_dim, self.out_dim)
        else:
            # map each token embedding to a patch reconstruction vector of size (out_c * ps * ps)
            ps = patch_size
            self.pix_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.out_dim * ps * ps),
            )
            self.patch_size = ps
            self.target_hw = (out_h, out_w)

        # Training state
        self.criterion = None
        self.optimizer = None
        self.to(self.device)

    # ---- training hooks ----
    def train_setup(self, prm):
        prm = prm or {}
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.to(self.device)
        # choose loss based on head
        if self.head_type == "class":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        if self.criterion is None or self.optimizer is None:
            self.train_setup(self.prm)
        self.train()
        for xb, yb in train_data:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            preds = self.forward(xb)
            loss = self.criterion(preds, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)                         # [B, C, H, W]
        tokens, grid_hw = self.patch(x)               # [B, N, D], (Hp,Wp)
        tokens = self.token_mlp(tokens)               # [B, N, D]

        if self.head_type == "class":
            # mean pool over tokens -> logits
            # pool expects [B, D, N]
            pooled = self.pool(tokens.transpose(1, 2)).squeeze(-1)  # [B, D]
            logits = self.cls_head(pooled)                          # [B, C]
            return logits

        # image-to-image: per-token -> unpatchify -> (optional) resize to target H,W
        per_token = self.pix_head(tokens)                           # [B, N, C*ps*ps]
        y = _unpatchify(per_token, grid_hw, self.out_dim, self.patch.patch_size)  # [B, C, H', W']

        tgt_h, tgt_w = self.target_hw
        if (y.shape[-2], y.shape[-1]) != (tgt_h, tgt_w):
            y = F.interpolate(y, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        return y
