import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------- Positional Encoding (batch_first) --------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal PE for [B, T, D]."""
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


# ----------------------------- CNN Encoder --------------------------------

class CNNEncoder(nn.Module):
    """Small from-scratch CNN; outputs memory [B, 1, D]."""
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, d_model, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(d_model)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(images)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)        # [B, D, H, W]
        x = self.pool(x).squeeze(-1).squeeze(-1)                 # [B, D]
        return x.unsqueeze(1)                                    # [B, 1, D]


# ----------------------- Transformer Decoder (BF) -------------------------

class TransformerDecoder(nn.Module):
    """Standard nn.TransformerDecoder with batch_first=True."""
    def __init__(self, vocab_size: int, d_model: int = 640, n_head: int = 8,
                 num_layers: int = 2, dim_ff: int = 2048, dropout: float = 0.2):
        super().__init__()
        assert d_model % n_head == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = self.embed.weight  # weight tying

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: [B, T]
        memory:     [B, S, D]  (S=1 here)
        returns logits [B, T, V]
        """
        x = self.embed(tgt_tokens) * math.sqrt(self.d_model)     # [B, T, D]
        x = self.pe(x)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.dec(tgt=x, memory=memory, tgt_mask=mask)        # [B, T, D]
        return self.proj(x)                                      # [B, T, V]


# --------------------------------- Net ------------------------------------

class Net(nn.Module):
    """
    CNN encoder → 1-token memory → Transformer decoder (batch_first).
    Returns a **Tensor**. Teacher forcing aligns time with labels for BLEU.
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        in_ch = int(in_shape[1])
        vocab = int(out_shape[0])

        d_model = int(prm.get("hidden_dim", 640))   # ≥640
        n_head  = int(prm.get("nhead", 8))          # divides d_model
        n_dec   = int(prm.get("dec_layers", 2))
        dim_ff  = int(prm.get("dim_ff", 2048))
        dropout = float(prm.get("dropout", 0.2))

        self.encoder = CNNEncoder(in_ch, d_model)
        self.decoder = TransformerDecoder(vocab, d_model, n_head, n_dec, dim_ff, dropout)
        self.vocab = vocab

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
        self.optimizer = None

    # ---- helpers ----
    @staticmethod
    def _norm_caps(caps: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if caps is None:
            return None
        if caps.ndim == 1:
            caps = caps.unsqueeze(0)
        elif caps.ndim == 3:
            caps = caps[:, 0, :]
        return caps.long()

    # ---- API ----
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get("lr", 1e-3)), 1e-3)  # floor to avoid freeze
        beta1 = min(0.99, max(0.7, float(prm.get("momentum", 0.9))))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)
            caps = self._norm_caps(captions)                    # [B, T]
            inp, tgt = caps[:, :-1], caps[:, 1:]                # [B, T-1]

            memory = self.encoder(images)                       # [B, 1, D]
            logits = self.decoder(inp, memory)                  # [B, T-1, V]

            # sentry
            assert images.dim() == 4
            assert logits.shape[1] == inp.shape[1]
            assert logits.shape[-1] == self.vocab

            loss = self.criterion(logits.reshape(-1, self.vocab), tgt.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        memory = self.encoder(images)                            # [B, 1, D]

        if captions is None:
            # simple greedy decode to return a logits tensor for generated seq
            B = images.size(0)
            sos_id, eos_id = 1, 2
            seq = torch.full((B, 1), sos_id, dtype=torch.long, device=self.device)
            max_len = 20
            for _ in range(max_len - 1):
                lg = self.decoder(seq, memory)                  # [B, t, V]
                nxt = lg[:, -1, :].argmax(-1, keepdim=True)     # [B, 1]
                seq = torch.cat([seq, nxt], dim=1)
                if (nxt == eos_id).all():
                    break
            return self.decoder(seq, memory)                     # [B, T, V]

        caps = self._norm_caps(captions).to(self.device)         # [B, T]
        inp = caps[:, :-1]                                       # [B, T-1]
        logits_step = self.decoder(inp, memory)                  # [B, T-1, V]
        # pad front so logits time matches labels T (BLEU-friendly)
        pad = torch.zeros((caps.size(0), 1, self.vocab), device=logits_step.device, dtype=logits_step.dtype)
        return torch.cat([pad, logits_step], dim=1)              # [B, T, V]
