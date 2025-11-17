import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        inputs: torch.Tensor,                             # [B, T] token ids
        hidden_state: Tuple[torch.Tensor, torch.Tensor],  # (h0, c0)
        features: Optional[torch.Tensor] = None           # unused, kept for API compatibility
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        emb = self.embedding(inputs)                      # [B, T, H]
        out, hidden_state = self.lstm(emb, hidden_state)  # [B, T, H]
        logits = self.linear(out)                         # [B, T, V]
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))

        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, self.hidden_size)

        self.rnn = LSTMDecoder(self.hidden_size, self.vocab_size)

        self.sos_idx = int(self.prm.get("sos", self.prm.get("sos_idx", 1)))
        self.eos_idx = int(self.prm.get("eos", 0))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.max_len = int(self.prm.get("max_length", 20))

        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    # ---------------- Training API ----------------
    def train_setup(self, prm):
        prm = prm or {}
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.train()
        self.to(self.device)

    def learn(self, train_data: Any):
        if self.optimizer is None or self.criterion is None:
            prm = getattr(train_data, "prm", self.prm)
            self.train_setup(prm)

        self.train()

        def _run_batch(images: torch.Tensor, captions: torch.Tensor):
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device).long()
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]

            if captions.size(1) <= 1:
                return

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            features = self._encode(images)  # [B, H]
            h0c0 = self.rnn.init_zero_hidden(images.size(0), self.device)
            logits, _ = self.rnn(inputs, h0c0, features)   # [B, T-1, V]

            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

        if isinstance(train_data, dict):
            images = train_data.get("images", train_data.get("x", None))
            captions = train_data.get("captions", train_data.get("y", None))
            if images is not None and captions is not None:
                _run_batch(images, captions)
        else:
            for batch in train_data:
                if isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        continue
                    images, captions = batch[0], batch[1]
                elif isinstance(batch, dict):
                    images = batch.get("images", batch.get("x", None))
                    captions = batch.get("captions", batch.get("y", None))
                else:
                    images = getattr(batch, "x", None)
                    captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue
                _run_batch(images, captions)

    # ---------------- Forward / Inference ----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        images = images.to(self.device, dtype=torch.float32)
        features = self._encode(images)

        if captions is not None:
            captions = captions.to(self.device).long()
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]
            if hidden_state is None:
                hidden_state = self.rnn.init_zero_hidden(images.size(0), self.device)
            logits, hidden_state = self.rnn(captions, hidden_state, features)
            return logits, hidden_state

        bsz = images.size(0)
        h = self.rnn.init_zero_hidden(bsz, self.device)
        seq = torch.full((bsz, 1), self.sos_idx, dtype=torch.long, device=self.device)

        for _ in range(self.max_len):
            step_logits, h = self.rnn(seq[:, -1:], h, features)
            next_tok = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        return seq, h

    # ---------------- Internals ----------------
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(images)      # [B, 512, 1, 1]
        x = x.flatten(1)          # [B, 512]
        x = self.fc(x)            # [B, H]
        return x

    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and isinstance(in_shape[0], int):
                return int(in_shape[0])          # (C,H,W)
            if len(in_shape) >= 4 and isinstance(in_shape[1], int):
                return int(in_shape[1])          # (B,C,H,W)
        return 3

    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
