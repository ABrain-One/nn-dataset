import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

def supported_hyperparameters():
    return {"lr", "momentum"}


class SimpleStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down = None
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.stem = SimpleStem(in_channels, 64)
        self.layer1 = ResBlock(64, 128, stride=2)
        self.layer2 = ResBlock(128, hidden_size, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        return x  # [B, S, H]


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, pad_idx: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(512, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=min(4 * hidden_size, 3072),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, memory: torch.Tensor, captions: torch.Tensor, hidden_state=None):
        b, s, h = memory.shape
        T = captions.size(1)
        pos_ids = torch.arange(T, device=captions.device).unsqueeze(0).expand(b, -1)
        emb = self.embedding(captions) + self.pos_emb(pos_ids)
        tgt_mask = self._causal_mask(T, emb.device)
        out = self.decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: Dict[str, Any], device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_layers = int(self.prm.get("num_layers", 2))
        self.num_heads = int(self.prm.get("num_heads", 8))
        self.vocab_size = self._first_int(out_shape)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 16))

        self.in_channels = self._infer_in_channels(in_shape)
        if self.hidden_size % self.num_heads != 0:
            self.hidden_size = max(self.num_heads, ((self.hidden_size // self.num_heads) + 1) * self.num_heads)

        self.encoder = ConvEncoder(self.in_channels, self.hidden_size)
        self.decoder = TransformerCaptionDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            pad_idx=self.pad_idx,
            dropout=float(self.prm.get("dropout", 0.1)),
        )

        self.cnn = self.encoder
        self.embedding = self.decoder.embedding
        self.fc_out = self.decoder.fc_out

        self.criterion = None
        self.criteria = None
        self.optimizer = None
        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and all(isinstance(v, int) for v in in_shape):
                return int(in_shape[0])
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])
        return 3

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        return int(x)

    def _normalize_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.dim() == 1:
            caps = caps.unsqueeze(0)
        elif caps.dim() == 3:
            caps = caps[:, 0, :]
        return caps

    def train_setup(self, prm: Dict[str, Any]):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train_setup(getattr(train_data, "prm", self.prm))
        for batch in train_data:
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue
                images, caps = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, caps = batch.get("x"), batch.get("y")
            else:
                images, caps = getattr(batch, "x", None), getattr(batch, "y", None)

            if images is None or caps is None:
                continue

            images = images.to(self.device)
            caps = self._normalize_caps(caps.to(self.device).long())
            if caps.size(1) <= 1:
                continue

            inp = caps[:, :-1]
            tgt = caps[:, 1:]

            mem = self.encoder(images)
            logits, _ = self.decoder(mem, inp)

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        mem = self.encoder(images.to(self.device))
        if captions is None:
            b = images.size(0)
            caps = torch.full((b, 1), self.sos_idx, dtype=torch.long, device=self.device)
            tokens = []
            for _ in range(self.max_len):
                logits, _ = self.decoder(mem, caps)
                step = logits[:, -1, :].argmax(-1, keepdim=True)
                tokens.append(step)
                caps = torch.cat([caps, step], dim=1)
                if (step == self.eos_idx).all():
                    break
            return torch.cat(tokens, dim=1), hidden_state

        caps = self._normalize_caps(captions.to(self.device).long())
        logits, hidden_state = self.decoder(mem, caps, hidden_state)
        return logits, hidden_state


def model_net(in_shape, out_shape, prm, device):
    return Net(in_shape, out_shape, prm, device)
