import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_zero_hidden(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        inputs: torch.Tensor,  # [B, T] token ids
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        features: Optional[torch.Tensor] = None,  # kept for API compatibility
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(inputs)                     # [B, T, H]
        out, hidden_state = self.lstm(emb, hidden_state) # [B, T, H]
        logits = self.linear(out)                        # [B, T, V]
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        # ---- API aliases / metadata ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # ---- sizes / tokens ----
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.sos_idx = int(self.prm.get("sos", 1))
        self.eos_idx = int(self.prm.get("eos", 0))
        self.max_len = int(self.prm.get("max_length", 20))

        # ---- Encoder: small CNN -> GAP -> FC ----
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, self.hidden_size)

        # ---- Decoder ----
        self.rnn = LSTMDecoder(self.hidden_size, self.vocab_size)

        # ---- training helpers (populated by train_setup) ----
        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------------- Training API ----------------
    def train_setup(self, prm):
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Any):
        """
        Accepts either:
          • an iterable of (images, captions) batches
          • a dict with tensors {'images': ..., 'captions': ...} for a single batch
        """
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)

        self.train()

        def _run_batch(images: torch.Tensor, captions: torch.Tensor):
            images = images.to(self.device).float()
            captions = captions.to(self.device).long()
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]  # [B, T]

            # need at least one next-token target
            if captions.size(1) <= 1:
                return

            inputs = captions[:, :-1]  # [B, T-1]
            targets = captions[:, 1:]  # [B, T-1]

            _features = self._encode(images)  # [B, H] (kept for API parity)

            h0c0 = self.rnn.init_zero_hidden(images.size(0), self.device)
            logits, _ = self.rnn(inputs, h0c0, _features)  # [B, T-1, V]

            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

        if isinstance(train_data, dict):
            _run_batch(train_data["images"], train_data["captions"])
        else:
            for images, captions in train_data:
                _run_batch(images, captions)

    # ---------------- Forward / Inference ----------------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        features = self._encode(images)  # [B, H]

        # Teacher forcing
        if captions is not None:
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]  # [B, T]
            if hidden_state is None:
                hidden_state = self.rnn.init_zero_hidden(images.size(0), self.device)
            logits, hidden_state = self.rnn(captions, hidden_state, features)  # [B, T, V]
            return logits, hidden_state

        # Greedy generation
        bsz = images.size(0)
        h = self.rnn.init_zero_hidden(bsz, self.device)
        seq = torch.full((bsz, 1), self.sos_idx, dtype=torch.long, device=self.device)  # [B, 1]

        for _ in range(self.max_len):
            step_logits, h = self.rnn(seq[:, -1:], h, features)  # feed last token
            next_tok = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        return seq, h

    # ---------------- Internals ----------------
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(images)     # [B, 512, 1, 1]
        x = x.flatten(1)         # [B, 512]
        x = self.fc(x)           # [B, H]
        return x

    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 4:
                return int(in_shape[1])  # (B, C, H, W)
            if len(in_shape) == 3:
                return int(in_shape[0])  # (C, H, W)
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
