import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        c = [64, 128, hidden_size // 2, hidden_size]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(c[0], c[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        if x.size(1) > 196:
            x = x[:, :196, :]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float, pad_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=min(3072, hidden_size * 4),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, memory: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(captions)
        emb = self.pos_enc(emb)
        t = captions.size(1)
        tgt_mask = torch.triu(
            torch.ones(t, t, device=captions.device, dtype=torch.bool),
            diagonal=1,
        )
        logits = self.decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.out(logits)
        return logits

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return z, z


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: Dict[str, Any], device: torch.device, *_ , **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 640))
        self.num_layers = int(self.prm.get("num_layers", 2))
        self.num_heads = int(self.prm.get("num_heads", 8))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 16))

        if self.hidden_size % self.num_heads != 0:
            self.hidden_size = max(self.num_heads, ((self.hidden_size // self.num_heads) + 1) * self.num_heads)

        self.encoder = ResNetEncoder(self.in_channels, self.hidden_size)
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=float(self.prm.get("dropout", 0.2)),
            pad_idx=self.pad_idx,
        )

        self.cnn = self.encoder.cnn
        self.embedding = self.decoder.embedding
        self.fc_out = self.decoder.out

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and isinstance(in_shape[0], int):
                return int(in_shape[0])
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])
        return 3

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and x:
            return Net._first_int(x[0])
        return int(x)

    def _normalize_captions(self, captions: torch.Tensor) -> torch.Tensor:
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3:
            captions = captions[:, 0, :]
        return captions

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
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.train_setup(getattr(train_data, "prm", self.prm))

        self.train()
        for batch in train_data:
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue
                images, captions = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("x", None)
                captions = batch.get("y", None)
                if images is None or captions is None:
                    continue
            else:
                images = getattr(batch, "x", None)
                captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue

            images = images.to(self.device)
            captions = self._normalize_captions(captions.to(self.device).long())
            if captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)
            logits = self.decoder(memory, inputs)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[Any] = None):
        memory = self.encoder(images.to(self.device))
        if captions is not None:
            captions = self._normalize_captions(captions.to(self.device).long())
            logits = self.decoder(memory, captions)
            return logits, hidden_state

        b = images.size(0)
        tokens = torch.full((b, 1), self.sos_idx, dtype=torch.long, device=self.device)
        for _ in range(self.max_len - 1):
            logits = self.decoder(memory, tokens)
            step_logits = logits[:, -1, :]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break
        return tokens

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
