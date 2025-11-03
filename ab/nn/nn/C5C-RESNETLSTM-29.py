import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------- Vision Transformer style encoder (compact, robust) ----------
class ViTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 768,
        patch_size: int = 16,
        depth: int = 4,
        nheads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding: [B, C, H, W] -> [B, D, H/ps, W/ps]
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    @staticmethod
    def _sinusoidal_pos_enc(n: int, d: int, device: torch.device) -> torch.Tensor:
        # [n, d]
        pe = torch.zeros(n, d, device=device)
        position = torch.arange(0, n, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, C, H, W]
        x = self.patch_embed(images)               # [B, D, H', W']
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, N, D] where N = H'*W'
        pos = self._sinusoidal_pos_enc(x.size(1), D, x.device)  # [N, D]
        x = x + pos.unsqueeze(0)                  # [B, N, D]
        x = self.encoder(x)                       # [B, N, D]
        feat = x.mean(dim=1)                      # [B, D] (mean-pooled token features)
        return feat


# ---------- LSTM Decoder (visual features concatenated to token embeddings) ----------
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size + hidden_size,  # token 768 + visual 768 = 1536
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        visual_feat: torch.Tensor,          # [B, 768]
        captions: torch.Tensor,             # [B, L] (teacher forcing input)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L = captions.shape
        tok = self.embedding(captions)      # [B, L, 768]
        vis = visual_feat.unsqueeze(1).expand(B, L, visual_feat.size(-1))  # [B, L, 768]
        x = torch.cat([tok, vis], dim=-1)   # [B, L, 1536]

        if hidden is None:
            hidden = self.init_zero_hidden(B, tok.device)

        out, hidden = self.rnn(x, hidden)   # [B, L, 768]
        logits = self.proj(out)             # [B, L, V]
        return logits, hidden


# ---------- Full Net ----------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Robust vocab size parsing (supports int, (V,), ((V,), ...))
        def _vsize(shape) -> int:
            if isinstance(shape, int):
                return int(shape)
            if isinstance(shape, (tuple, list)) and len(shape) > 0:
                first = shape[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    return int(first[0])
                return int(first)
            return int(shape)

        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3
        self.vocab_size = _vsize(out_shape)
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.patch_size = int(prm.get("patch_size", 16))
        self.num_layers = int(prm.get("num_layers", 1))
        self.dropout = float(prm.get("dropout", 0.1))

        self.encoder = ViTEncoder(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_size,
            patch_size=self.patch_size,
            depth=max(1, int(prm.get("vit_depth", 4))),
            nheads=max(1, int(prm.get("vit_heads", 8))),
            dropout=self.dropout,
        )
        self.decoder = LSTMDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.criteria: Optional[Tuple[nn.Module]] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # --- training helpers ---
    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def init_zero_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.init_zero_hidden(batch_size, self.device)

    # --- forward ---
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        images = images.to(self.device)
        vis = self.encoder(images)  # [B, hidden_size]

        if captions is None:
            # encoder-only path
            return vis, None

        captions = captions.to(self.device)  # [B, L]
        logits, hidden_state = self.decoder(vis, captions, hidden_state)
        return logits, hidden_state

    # --- simple training loop (teacher forcing) ---
    def learn(self, train_data):
        if self.criteria is None or self.optimizer is None:
            # fallback if user forgot to call train_setup
            self.train_setup({})

        criterion = self.criteria[0]
        self.train()
        for images, captions in train_data:
            # Teacher forcing: predict next token
            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            logits, _ = self.forward(images, inp)  # [B, L-1, V]
            loss = criterion(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
