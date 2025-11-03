from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------------- Encoder ----------------
class SELayer2d(nn.Module):
    """Squeeze-and-Excitation for 2D feature maps."""
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w


class SpatialAttentionEncoder(nn.Module):
    """
    Simple CNN -> [B, S, H] token sequence.
    Outputs tokens of dimension `hidden_dim` (>=640 recommended by harnesses).
    """
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768), nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 1x1 conv instead of Linear for channel remapping on 4D tensors
        self.proj = nn.Conv2d(768, hidden_dim, kernel_size=1, bias=True)
        self.se = SELayer2d(hidden_dim, reduction=8)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 3, H, W]
        x = self.cnn(x)                 # [B, 768, H', W']
        x = self.proj(x)                # [B, hidden, H', W']
        x = self.se(x)                  # SE reweight
        x = F.relu(self.bn(x), inplace=True)
        x = self.do(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, S, H]
        return x


# ---------------- Decoder ----------------
class EnhancedTransformerDecoder(nn.Module):
    """
    Transformer decoder working on tokenized image memory.
    """
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 768,   # memory/hidden size
        hidden_size: int = 768,   # decoder d_model
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert hidden_size == feature_dim, "feature_dim must equal hidden_size for decoder"
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=min(hidden_size * 2, 3072),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.do = nn.Dropout(dropout)

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        # [T, T] mask with True above diagonal
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        features: Tensor,          # [B, S, H]
        captions: Tensor,          # [B, T] (teacher forcing)
        key_padding_mask: Optional[Tensor] = None,  # [B, S], True => ignore
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Embed target tokens
        emb = self.do(self.embedding(captions))      # [B, T, H]
        tgt_mask = self._causal_mask(emb.size(1), emb.device)
        out = self.decoder(tgt=emb, memory=features, tgt_mask=tgt_mask, memory_key_padding_mask=key_padding_mask)
        logits = self.fc(out)                        # [B, T, V]
        return logits, None

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return z, z


# ---------------- Net wrapper ----------------
class Net(nn.Module):
    """
    Expected harness API:
      __init__(in_shape, out_shape, prm, device)
      train_setup(prm)
      learn(train_data)
      forward(images, captions=None, hidden_state=None) -> (logits, hidden)
      supported_hyperparameters() (static or module-level)
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: Dict[str, float], device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_heads = int(self.prm.get("num_heads", 8))
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        self.vocab_size = self._first_int(out_shape)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))

        self.encoder = SpatialAttentionEncoder(hidden_dim=self.hidden_size, dropout=float(self.prm.get("dropout", 0.1)))
        self.decoder = EnhancedTransformerDecoder(
            vocab_size=self.vocab_size,
            feature_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=int(self.prm.get("num_layers", 3)),
            num_heads=self.num_heads,
            dropout=float(self.prm.get("decoder_dropout", 0.1)),
        )

        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    # ---- training helpers ----
    def train_setup(self, prm: Dict[str, float]):
        prm = prm or {}
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        # Fall back to self.prm if loader doesn't carry prm
        self.train_setup(getattr(train_data, "prm", self.prm))
        self.train()

        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
            if captions.size(1) <= 1:
                continue

            # teacher forcing: predict token t given 0..t-1
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ---- forward ----
    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        hidden_state: Optional[Any] = None,
    ) -> Tuple[Tensor, Any]:
        # Encode image to memory tokens
        memory = self.encoder(images.to(self.device))  # [B, S, H]

        # If no captions given, do single-step from SOS
        if captions is None:
            captions = torch.full((images.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)

        logits, _ = self.decoder(memory, captions.long().to(self.device))
        return logits, hidden_state

    # ---- utils ----
    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        return int(x)
