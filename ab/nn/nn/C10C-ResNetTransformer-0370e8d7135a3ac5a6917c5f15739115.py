import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}




class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(4, c // r))
        self.fc2 = nn.Linear(max(4, c // r), c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        s = x.mean(dim=(2, 3))               # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))       # [B, C]
        return x * s.view(b, c, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        self.skip = None
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        out = F.relu(out + identity, inplace=True)
        out = self.se(out)
        return out


class PositionalEncodingBF(nn.Module):
    """Sinusoidal PE for batch_first tensors [B, T, D]."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        T = x_btd.size(1)
        return x_btd + self.pe[:T].unsqueeze(0)


# ----------------------------- Encoder (fast start) ---------------------------

class CNNEncoder(nn.Module):
    """
    Lightweight CNN that learns quickly from scratch, then projects to D.
    Outputs memory as [B, S, D] with S=1 (a single global token).
    """
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = ConvBlock(64, 128, stride=2)   # 1/8
        self.stage2 = ConvBlock(128, 256, stride=2)  # 1/16
        self.stage3 = ConvBlock(256, 256, stride=1)  # keep
        self.head = nn.Sequential(
            nn.Conv2d(256, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                 # [B, D, 1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x).squeeze(-1).squeeze(-1)     # [B, D]
        return x.unsqueeze(1)                        # [B, 1, D] single memory token


# --------------------------- Transformer Decoder (BF) -------------------------

class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 640, nhead: int = 8,
                 num_layers: int = 2, dim_ff: int = 2048, dropout: float = 0.2):
        super().__init__()
        assert d_model % nhead == 0
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncodingBF(d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,               # keep batch_first to avoid shape pitfalls
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying helps early quality
        self.proj.weight = self.embedding.weight

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(
        self,
        inputs: torch.Tensor,                  # [B, T]
        features: torch.Tensor                 # [B, S=1, D]
    ) -> torch.Tensor:
        assert features.dim() == 3, "features must be [B, S, D]"
        x = self.embedding(inputs) * math.sqrt(self.d_model)     # [B, T, D]
        x = self.pos(x)                                          # [B, T, D]
        T = x.size(1)
        tgt_mask = self._causal_mask(T, x.device)                # [T, T]
        dec = self.decoder(tgt=x, memory=features, tgt_mask=tgt_mask)  # [B, T, D]
        logits = self.proj(dec)                                  # [B, T, V]
        return logits


# ------------------------------------ Net ------------------------------------

class Net(nn.Module):
    """
    Fast-start CNN encoder (with SE) → 1-token memory → Transformer decoder (batch_first).
    Returns **logits Tensor only** to match BLEU metric expectations.
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device

        in_channels = int(in_shape[1])
        vocab_size = int(out_shape[0])

        # Reasonable defaults that train quickly
        d_model = int(prm.get("hidden_dim", 640))      # ≥640 per your rule
        nhead = int(prm.get("nhead", 8))               # divides d_model (640 % 8 == 0)
        dec_layers = int(prm.get("dec_layers", 2))
        dim_ff = int(prm.get("dim_ff", 2048))
        dropout = float(prm.get("dropout", 0.2))

        self.encoder = CNNEncoder(in_ch=in_channels, d_model=d_model)
        self.rnn = TransformerCaptionDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead,
            num_layers=dec_layers, dim_ff=dim_ff, dropout=dropout
        )
        self.vocab_size = vocab_size

        # Loss/optim set in train_setup
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
        self.optimizer = None

    # ---------- helpers ----------

    @staticmethod
    def _normalize_caps(caps: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if caps is None:
            return None
        if caps.ndim == 1:
            caps = caps.unsqueeze(0)
        elif caps.ndim == 3:                 # [B, 1, T] → [B, T]
            caps = caps[:, 0, :]
        return caps.long()

    # ---------- required API ----------

    def train_setup(self, prm: dict):
        self.to(self.device)
        # Floor LR so DF-randomized very low LRs don’t freeze training
        lr = max(float(prm.get("lr", 1e-3)), 1e-3)
        beta1 = float(prm.get("momentum", 0.9))
        beta1 = min(0.99, max(0.7, beta1))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            captions = self._normalize_caps(captions)            # [B, T]
            # Teacher forcing
            inputs = captions[:, :-1]                             # [B, T-1]
            targets = captions[:, 1:]                             # [B, T-1]

            memory = self.encoder(images)                         # [B, 1, D]
            logits = self.rnn(inputs, memory)                     # [B, T-1, V]

            # Sentry checks
            assert images.dim() == 4
            assert logits.shape[1] == inputs.shape[1]
            assert logits.shape[-1] == self.vocab_size

            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def _greedy_generate(self, memory: torch.Tensor, max_len: int = 20) -> torch.Tensor:
        """Simple greedy decode to support captions=None path."""
        B = memory.size(0)
        start_id, eos_id = 1, 2
        seq = torch.full((B, 1), start_id, dtype=torch.long, device=self.device)
        for _ in range(max_len - 1):
            logits = self.rnn(seq, memory)                        # [B, t, V]
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return seq

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state=None
    ) -> torch.Tensor:
        """
        Returns **logits Tensor**.
        - If captions is given (teacher forcing): logits shape **[B, T, V]** to match labels time length for BLEU.
          We compute model outputs for T-1 positions and pad a dummy first step so BLEU sees aligned sequences.
        - If captions is None: run a short greedy decode and return logits for the generated sequence.
        """
        images = images.to(self.device, non_blocking=True)
        memory = self.encoder(images)  # [B, 1, D]

        if captions is None:
            seq = self._greedy_generate(memory, max_len=20)      # [B, T]
            logits = self.rnn(seq, memory)                       # [B, T, V]
            return logits

        caps = self._normalize_caps(captions).to(self.device)    # [B, T]
        inputs = caps[:, :-1]                                    # [B, T-1]
        logits_step = self.rnn(inputs, memory)                    # [B, T-1, V]

        # Pad at the front so time length matches labels (T)
        pad = torch.zeros((caps.size(0), 1, self.vocab_size), device=logits_step.device, dtype=logits_step.dtype)
        logits = torch.cat([pad, logits_step], dim=1)            # [B, T, V]
        return logits
