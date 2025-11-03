import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Iterable


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    """
    Minimal, compile-ready image-encoder + caption-decoder.
    - Encoder: small CNN -> GAP -> Linear to hidden_size
    - Decoder: Embedding + 1-layer LSTM -> Linear to vocab
    """
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: dict, device: torch.device, *_, **__) -> None:
        super().__init__()
        self.device = device

        # ---- robust shape parsing ----
        self.in_channels = int(in_shape[1]) if len(in_shape) >= 2 else int(in_shape[0])

        def _first_int(x: Any) -> int:
            while isinstance(x, (tuple, list)):
                x = x[0]
            return int(x)

        self.vocab_size = _first_int(out_shape)

        # ---- config ----
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.num_layers = 1
        self.dropout = float(prm.get("dropout", 0.1))

        # ---- encoder (CNN -> 1x1 -> linear) ----
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.enc_proj = nn.Linear(256, self.hidden_size)

        # ---- decoder (Embedding + LSTM + Linear) ----
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.readout = nn.Linear(self.hidden_size, self.vocab_size)
        self.dec_dropout = nn.Dropout(self.dropout)

        self.to(self.device)

    # -------------------- training helpers --------------------
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-4)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        """Simple teacher-forced training loop over (images, captions)."""
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            logits, _ = self.forward(images, captions)
            # Predict next token => align with captions[:, 1:]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), captions[:, 1:].reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            yield loss.detach().item()

    # -------------------- forward --------------------
    def init_zero_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)                 # [B, 256]
        feats = self.enc_proj(feats)                 # [B, H]
        return feats

    def forward(self, images: torch.Tensor, captions: torch.Tensor | None = None, hidden_state=None):
        """
        If captions is provided: teacher forcing (returns [B, T-1, V]).
        If captions is None: greedy decode up to max_len (returns [B, max_len, V]).
        """
        B = images.size(0)
        feats = self._encode(images)                 # [B, H]

        if hidden_state is None:
            hidden_state = self.init_zero_hidden(B, feats.device)
        h, c = hidden_state

        if captions is not None:
            # Teacher forcing
            inputs = captions[:, :-1]               # [B, T-1]
            emb = self.embedding(inputs)            # [B, T-1, H]
            # Add global image context by addition
            emb = emb + feats.unsqueeze(1)          # broadcast add
            out, (h, c) = self.lstm(emb, (h, c))    # [B, T-1, H]
            logits = self.readout(self.dec_dropout(out))  # [B, T-1, V]
            return logits, (h, c)

        # Greedy decoding
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=feats.device)
        logits_steps = []
        for _ in range(self.max_len):
            emb = self.embedding(tokens[:, -1:])    # [B, 1, H]
            emb = emb + feats.unsqueeze(1)          # add image context
            out, (h, c) = self.lstm(emb, (h, c))    # [B, 1, H]
            step_logits = self.readout(out)         # [B, 1, V]
            logits_steps.append(step_logits)
            next_tok = step_logits.argmax(dim=-1)   # [B, 1]
            tokens = torch.cat([tokens, next_tok], dim=1)

        logits = torch.cat(logits_steps, dim=1)     # [B, max_len, V]
        return logits, (h, c)
