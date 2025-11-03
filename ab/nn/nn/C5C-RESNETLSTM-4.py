import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from torch import Tensor


# ---------------------------------------------------------------------
# Public API hook kept from your file
# ---------------------------------------------------------------------
def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _first_int(x: Any) -> int:
    """Safely extract an int (e.g., vocab size) from possibly nested tuples/lists."""
    if isinstance(x, int):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _first_int(x[0])
    try:
        return int(x)
    except Exception:
        return 10000  # safe fallback


def _infer_in_channels(in_shape: Any) -> int:
    """
    Infer channel count from in_shape. Works for (C,H,W) or (N,C,H,W).
    Defaults to 3 if ambiguous.
    """
    if isinstance(in_shape, (tuple, list)):
        if len(in_shape) >= 1 and isinstance(in_shape[0], int):
            # shape like (C,H,W)
            return int(in_shape[0])
        if len(in_shape) >= 2 and isinstance(in_shape[1], int):
            # shape like (N,C,H,W)
            return int(in_shape[1])
    return 3


def _subsequent_mask(T: int, device: torch.device) -> Tensor:
    """Causal mask for autoregressive decoding."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


# ---------------------------------------------------------------------
# CBAM / SE blocks
# ---------------------------------------------------------------------
class SqueezeExcitation_CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4, device: Optional[torch.device] = None, cbam: bool = True):
        super().__init__()
        self.reduction = max(1, int(reduction))
        self.device = device
        self.cbam = cbam

        if self.cbam:
            # Channel attention (MLP w/ conv1x1) + Spatial attention (conv7x7)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, max(1, in_channels // self.reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, in_channels // self.reduction), in_channels, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
            self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
            self.spatial_sigmoid = nn.Sigmoid()
        else:
            # Standard SE
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, max(1, in_channels // self.reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, in_channels // self.reduction), in_channels, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.cbam:
            # Channel attention
            ca = self.channel_attention(x)                # [B,C,1,1]
            x = x * ca

            # Spatial attention
            sa = self.spatial_sigmoid(self.spatial_attention(x))  # [B,1,H,W]
            x = x * sa
            return x
        else:
            weight = self.attention(x)  # [B,C,1,1]
            return x * weight


class ModifiedBagNetUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, cbam: bool = True):
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.identity_conv = None
        if self.resize_identity:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

        self.cbam_att = SqueezeExcitation_CBAM(out_channels, cbam=cbam) if cbam else None
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.resize_identity:
            identity = self.identity_conv(identity)

        out = self.body(x)

        # Align spatial sizes if needed
        if out.size()[2:] != identity.size()[2:]:
            identity = F.interpolate(identity, size=out.shape[-2:], mode="nearest")

        out = out + identity
        if self.cbam_att:
            out = self.cbam_att(out)
        return self.activ(out)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class Net(nn.Module):
    """
    CNN encoder (with CBAM/SE blocks) → Transformer (default) or LSTM decoder.
    """
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Hyperparameters / options
        self.use_cbam = bool(prm.get("use_cbam", True))
        self.use_transformer = bool(prm.get("use_transformer", True))
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.num_heads = int(prm.get("num_heads", 8))
        self.dropout = float(prm.get("dropout", 0.1))
        self.d_ff = int(prm.get("d_ff", 2048))
        self.max_len = int(prm.get("max_len", 256))

        # I/O sizes
        self.vocab_size = _first_int(out_shape)
        in_channels = _infer_in_channels(in_shape)

        # ---------------- Encoder ----------------
        # Backbone that ends with C = hidden_size//4; then project to hidden_size
        hs = self.hidden_size
        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(in_channels, hs, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ModifiedBagNetUnit(hs, hs, kernel_size=3, stride=1, cbam=self.use_cbam),
            ModifiedBagNetUnit(hs, hs // 2, kernel_size=1, stride=1, cbam=False),
            ModifiedBagNetUnit(hs // 2, hs // 4, kernel_size=3, stride=2, cbam=False),
        )
        self.enc_out_proj = nn.Conv2d(hs // 4, hs, kernel_size=1, bias=False)

        # ---------------- Decoder ----------------
        if self.use_transformer:
            # Token + positional embeddings
            self.embed_tokens = nn.Embedding(self.vocab_size, hs, padding_idx=0)
            self.pos_embed = nn.Embedding(self.max_len, hs)
            self.tgt_dropout = nn.Dropout(self.dropout)

            # Standard PyTorch TransformerDecoder
            layer = nn.TransformerDecoderLayer(
                d_model=hs,
                nhead=self.num_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=1, norm=None)
        else:
            # LSTM fallback
            self.embed_tokens = nn.Embedding(self.vocab_size, hs, padding_idx=0)
            self.decoder = nn.LSTM(hs, hs, batch_first=True)

        # Projection head (kept close to your original idea)
        self.projector = nn.Linear(hs, 640)
        self.fc_final = nn.Linear(640, self.vocab_size)

        # Train-time objects (set in train_setup)
        self.criteria: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ------------- Required hooks -------------
    def train_setup(self, prm: dict):
        self.to(self.device)
        self.train()
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data):
        self.train_setup(getattr(train_data, "prm", {}))
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.long().to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
            if captions.size(1) <= 1:
                continue

            logits, _ = self.forward(images, captions=captions)  # [B,T-1,V]
            targets = captions[:, 1:]  # teacher forcing target
            loss = self.criteria(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ------------- Public helpers -------------
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        h0 = torch.zeros(batch, self.hidden_size, device=device)
        c0 = torch.zeros(batch, self.hidden_size, device=device)
        return h0, c0

    # ------------- Core forward -------------
    def _encode(self, images: Tensor) -> Tensor:
        """
        Encode images to memory tokens for the decoder.
        Returns memory as [B, P, H].
        """
        feats = self.encoder_backbone(images)              # [B, C', H', W']  (C' = hidden_size//4)
        feats = self.enc_out_proj(feats)                   # [B, hidden_size, H', W']
        B, C, H, W = feats.shape
        mem = feats.flatten(2).transpose(1, 2)             # [B, P=H'*W', hidden_size]
        return mem

    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Training:
            returns (logits[B, T-1, V], (h,c))
        Inference (captions=None):
            returns (logits[B, V] for first BOS step, (h,c))
        """
        images = images.to(self.device)
        mem = self._encode(images)  # [B, P, H]

        if captions is not None:
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            captions = captions.long().to(self.device)

            # Teacher-forcing: predict next token
            inputs = captions[:, :-1]  # [B, T-1]
            B, T = inputs.shape[0], inputs.shape[1]

            if self.use_transformer:
                # Embedding + simple learned positional enc
                positions = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)  # [B,T]
                tgt = self.embed_tokens(inputs) + self.pos_embed(positions)                # [B,T,H]
                tgt = self.tgt_dropout(tgt)

                tgt_mask = _subsequent_mask(T, self.device)  # [T,T] bool
                out = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)                 # [B,T,H]
            else:
                # LSTM decode (full sequence, teacher forcing)
                tgt = self.embed_tokens(inputs)                                              # [B,T,H]
                out, hidden_state = self.decoder(tgt, hidden_state)                          # [B,T,H]

            logits = self.fc_final(self.projector(out))  # [B,T,V]
            return logits, (torch.zeros(B, self.hidden_size, device=self.device),
                            torch.zeros(B, self.hidden_size, device=self.device))

        # Inference: single BOS step
        B = images.size(0)
        bos = torch.full((B, 1), 1, dtype=torch.long, device=self.device)  # BOS=1, PAD=0
        if self.use_transformer:
            positions = torch.zeros_like(bos)
            tgt = self.embed_tokens(bos) + self.pos_embed(positions)       # [B,1,H]
            out = self.decoder(tgt=tgt, memory=mem)                         # [B,1,H]
        else:
            tgt = self.embed_tokens(bos)                                    # [B,1,H]
            out, hidden_state = self.decoder(tgt, hidden_state)             # [B,1,H]

        logits = self.fc_final(self.projector(out))                         # [B,1,V]
        return logits.squeeze(1), (torch.zeros(B, self.hidden_size, device=self.device),
                                   torch.zeros(B, self.hidden_size, device=self.device))


# Optional factory some harnesses expect
def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
