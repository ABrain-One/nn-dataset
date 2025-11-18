import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

def supported_hyperparameters():
    return {"lr", "momentum"}


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_size, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.view(x.size(0), -1)  # [B, H]


class GRUCaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, pad_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def init_hidden_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return feat.unsqueeze(0)  # [1, B, H]

    def forward(self, feat: torch.Tensor, captions: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        if hidden is None:
            hidden = self.init_hidden_from_feat(feat)
        emb = self.embedding(captions)
        out, hidden = self.gru(emb, hidden)
        logits = self.fc_out(out)
        return logits, hidden


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: Dict[str, Any], device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.vocab_size = self._first_int(out_shape)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 16))

        self.in_channels = self._infer_in_channels(in_shape)

        self.encoder = ConvEncoder(self.in_channels, self.hidden_size)
        self.decoder = GRUCaptionDecoder(self.vocab_size, self.hidden_size, self.pad_idx)

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

            feat = self.encoder(images)
            logits, _ = self.decoder(feat, inp)

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        feat = self.encoder(images.to(self.device))
        if captions is None:
            b = images.size(0)
            prev = torch.full((b, 1), self.sos_idx, dtype=torch.long, device=self.device)
            tokens = []
            hidden = None
            for _ in range(self.max_len):
                logits, hidden = self.decoder(feat, prev, hidden)
                step = logits[:, -1, :].argmax(-1, keepdim=True)
                tokens.append(step)
                prev = step
                if (step == self.eos_idx).all():
                    break
            return torch.cat(tokens, dim=1), hidden

        caps = self._normalize_caps(captions.to(self.device).long())
        logits, hidden_state = self.decoder(feat, caps, hidden_state)
        return logits, hidden_state


def model_net(in_shape, out_shape, prm, device):
    return Net(in_shape, out_shape, prm, device)
