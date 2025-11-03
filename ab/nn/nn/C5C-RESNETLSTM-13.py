import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Any


def supported_hyperparameters() -> set:
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- Basic shape info ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = (
            in_shape[1] if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3
        )
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))

        # ---- Encoder (CNN) ----
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Ensure a fixed 4x4 spatial size before the final projection
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.final_proj = nn.Linear(512 * 4 * 4, self.hidden_size)

        # ---- Decoder (LSTM) ----
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        # Input size must match embedding size
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        # Defaults for special tokens if used during inference
        self.sos_idx = int(self.prm.get("sos", 1))
        self.eos_idx = int(self.prm.get("eos", 0))

        # Training helpers (populated in train_setup)
        self.criteria = None
        self.criterion = None
        self.optimizer = None

        self.to(self.device)

    # ----------------- API helpers -----------------
    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Any):
        # Placeholder training loop (kept minimal to match original intent)
        self.train()
        if self.optimizer is None:
            self.train_setup(self.prm)
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]
            if captions.dim() != 2 or captions.size(1) <= 1:
                continue
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.forward(images, inputs)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ----------------- Forward -----------------
    def forward(
        self, images: torch.Tensor, captions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode images
        feats = self.cnn(images)
        feats = self.adapt_pool(feats)          # [B, 512, 4, 4]
        feats = feats.flatten(1)                # [B, 512*4*4]
        feats = self.final_proj(feats)          # [B, hidden_size] (kept for potential future use)

        if captions is not None:
            # Teacher forcing path
            embedded = self.embedding(captions)             # [B, T, H]
            output, (h_n, c_n) = self.rnn(embedded, None)   # [B, T, H]
            logits = self.fc_out(output)                    # [B, T, V]
            return logits, h_n
        else:
            # One-step greedy decode (kept minimal as in original)
            B = images.size(0)
            h0 = torch.zeros(1, B, self.hidden_size, device=self.device)
            c0 = torch.zeros(1, B, self.hidden_size, device=self.device)
            start_tok = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
            embedded = self.embedding(start_tok)            # [B, 1, H]
            output, (h_n, c_n) = self.rnn(embedded, (h0, c0))
            logits = self.fc_out(output)                    # [B, 1, V]
            return logits[:, 0, :], h_n

    # ----------------- Utility -----------------
    @staticmethod
    def _first_int(x) -> int:
        """Extract an int from possibly nested out_shape tuples/lists."""
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)
