# new_nn.py
import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, Optional, Tuple

# --- Keep CNBlockConfig usage, but allow running if torchvision isn't present ---
try:
    from torchvision.models.convnext import CNBlockConfig  # type: ignore
except Exception:
    class CNBlockConfig:  # minimal fallback so this file is importable
        def __init__(self, in_ch: int, out_ch: Optional[int], num_layers: int, *_, **__):
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.num_layers = num_layers


# Properly closed list; no function definitions inside it.
block_setting = [
    CNBlockConfig(96, 192, 3),
    CNBlockConfig(192, 384, 3),
    CNBlockConfig(384, 768, 27),
    CNBlockConfig(768, None, 3),
]


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------------- Minimal, safe model with the required API ---------------- #

class _Encoder(nn.Module):
    """Lightweight CNN encoder -> single feature vector per image."""
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).flatten(1)     # [B, 256]
        return self.fc(x)               # [B, H]


class _LSTMDecoder(nn.Module):
    """Simple token decoder with embedding + LSTM."""
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def init_zero_hidden(self, batch: int, device: torch.device):
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    def forward(self, inputs: torch.Tensor, hidden, _features: Optional[torch.Tensor] = None):
        x = self.embed(inputs)          # [B, T, H]
        y, hidden = self.lstm(x, hidden)
        logits = self.proj(y)           # [B, T, V]
        return logits, hidden


class Net(nn.Module):
    """
    Required API:
      - __init__(in_shape, out_shape, prm, device, *_, **__)
      - train_setup(prm)
      - learn(train_data)
      - forward(images, captions=None, hidden_state=None)
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.sos = int(self.prm.get("sos", 1))
        self.eos = int(self.prm.get("eos", 0))
        self.max_len = int(self.prm.get("max_length", 20))

        self.encoder = _Encoder(self.in_channels, self.hidden_size)
        self.decoder = _LSTMDecoder(self.hidden_size, self.vocab_size)

        # map image features -> initial LSTM state
        self.h_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_init = nn.Linear(self.hidden_size, self.hidden_size)

        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------------- training plumbing ----------------
    def train_setup(self, prm: Dict[str, Any]):
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]] | Dict[str, torch.Tensor]):
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)
        self.train()

        def _run(images: torch.Tensor, captions: torch.Tensor):
            images = images.to(self.device).float()
            captions = captions.to(self.device).long()
            if captions.size(1) <= 1:
                return

            inputs = captions[:, :-1]    # teacher forcing
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

        if isinstance(train_data, dict):
            _run(train_data["images"], train_data["captions"])
        else:
            for images, captions in train_data:
                _run(images, captions)

    # ---------------- forward / inference ----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        B = images.size(0)
        feats = self.encoder(images.to(self.device).float())              # [B, H]
        if hidden_state is None:
            h0 = torch.tanh(self.h_init(feats)).unsqueeze(0)              # [1, B, H]
            c0 = torch.tanh(self.c_init(feats)).unsqueeze(0)              # [1, B, H]
            hidden_state = (h0, c0)

        if captions is not None:
            logits, hidden_state = self.decoder(captions.to(self.device).long(), hidden_state, feats)
            return logits, hidden_state

        # Greedy decode
        seq = torch.full((B, 1), self.sos, dtype=torch.long, device=self.device)
        h = hidden_state
        for _ in range(self.max_len):
            step_logits, h = self.decoder(seq[:, -1:], h, feats)
            next_tok = step_logits[:, -1].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos).all():
                break
        return seq, h

    # ---------------- helpers ----------------
    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 4:
                return int(in_shape[1])    # (B, C, H, W)
            if len(in_shape) == 3:
                return int(in_shape[0])    # (C, H, W)
        return 3

    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)):
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)
