import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------------- blocks ----------------

class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        m = max(4, c // r)
        self.fc1 = nn.Linear(c, m)
        self.fc2 = nn.Linear(m, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


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
            self.skip = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        if self.skip is not None:
            id = self.skip(id)
        x = F.relu(x + id, inplace=True)
        x = self.se(x)
        return x


class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------------- encoder/decoder ----------------

class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.s1 = ConvBlock(64, 128, stride=2)
        self.s2 = ConvBlock(128, 256, stride=2)
        self.s3 = ConvBlock(256, 256, stride=1)
        self.head = nn.Sequential(
            nn.Conv2d(256, d_model, 1, bias=False), nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x); x = self.s1(x); x = self.s2(x); x = self.s3(x)
        x = self.head(x).squeeze(-1).squeeze(-1)        # [B, D]
        return x.unsqueeze(1)                           # [B, 1, D]


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab: int, d_model: int = 640, nhead: int = 8, layers: int = 2, dim_ff: int = 2048, dropout: float = 0.2):
        super().__init__()
        assert d_model % nhead == 0
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pe = PositionalEncodingBF(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.proj = nn.Linear(d_model, vocab, bias=False)
        self.proj.weight = self.embed.weight

    @staticmethod
    def _causal_mask(T: int, device: torch.device):
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) * math.sqrt(self.embed.embedding_dim)
        x = self.pe(x)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.dec(tgt=x, memory=memory, tgt_mask=mask)
        return self.proj(x)                                   # [B, T, V]


# ---------------- Net (API) ----------------

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        in_ch = int(in_shape[1])
        vocab = int(out_shape[0])

        d_model = int(prm.get("hidden_dim", 640))
        nhead = int(prm.get("nhead", 8))
        layers = int(prm.get("dec_layers", 2))
        dim_ff = int(prm.get("dim_ff", 2048))
        dropout = float(prm.get("dropout", 0.2))

        self.encoder = CNNEncoder(in_ch, d_model)
        self.rnn = TransformerCaptionDecoder(vocab, d_model, nhead, layers, dim_ff, dropout)
        self.vocab = vocab

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
        self.optimizer = None

    @staticmethod
    def _norm_caps(caps: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if caps is None: return None
        if caps.ndim == 1: caps = caps.unsqueeze(0)
        elif caps.ndim == 3: caps = caps[:, 0, :]
        return caps.long()

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get("lr", 1e-3)), 1e-3)
        b1 = min(0.99, max(0.7, float(prm.get("momentum", 0.9))))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(b1, 0.999), weight_decay=1e-4)
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            caps = self._norm_caps(captions)                 # [B, T]
            inp, tgt = caps[:, :-1], caps[:, 1:]             # [B, T-1]

            mem = self.encoder(images)                       # [B, 1, D]
            logits = self.rnn(inp, mem)                      # [B, T-1, V]

            assert logits.shape[1] == inp.shape[1] and logits.shape[-1] == self.vocab
            loss = self.criterion(logits.reshape(-1, self.vocab), tgt.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        mem = self.encoder(images)                           # [B, 1, D]

        if captions is None:
            # simple greedy stub
            B = images.size(0)
            seq = torch.full((B, 1), 1, dtype=torch.long, device=self.device)  # <SOS>=1
            for _ in range(19):
                lg = self.rnn(seq, mem)
                nxt = lg[:, -1, :].argmax(-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
                if (nxt == 2).all(): break                   # <EOS>=2
            return self.rnn(seq, mem)

        caps = self._norm_caps(captions).to(self.device)     # [B, T]
        inp = caps[:, :-1]                                   # [B, T-1]
        logits = self.rnn(inp, mem)                          # [B, T-1, V]
        pad = torch.zeros((caps.size(0), 1, self.vocab), device=logits.device, dtype=logits.dtype)
        return torch.cat([pad, logits], dim=1)               # [B, T, V]
