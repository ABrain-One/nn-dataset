import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    """
    Minimal, compile-ready image-encoder + caption-decoder.
    Encoder: small CNN -> pooled -> linear to hidden size
    Decoder: Embedding -> LSTM -> Linear to vocab
    """

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Robust shape inference
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._infer_vocab_size(out_shape)

        # Hyperparameters
        prm = prm or {}
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_idx = int(prm.get("sos_idx", 1))

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
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

        # ----- Decoder -----
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(self.device)

    # -------- Utilities --------
    @staticmethod
    def _infer_in_channels(in_shape) -> int:
        try:
            return int(in_shape[1])
        except Exception:
            return 3

    @staticmethod
    def _infer_vocab_size(out_shape) -> int:
        x = out_shape
        while isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError("Invalid out_shape: empty container.")
            x = x[0]
        return int(x)

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Return zero (h, c) for LSTM (API compatibility)."""
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    # -------- Training helpers --------
    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        """Standard teacher-forcing training loop over (images, captions)."""
        self.train()
        last_loss = 0.0
        for images, captions in train_data:
            images = images.to(self.device)               # (B, C, H, W)
            captions = captions.to(self.device)           # (B, T) or (B, 1, T)
            if captions.dim() == 3:
                captions = captions[:, 0, :]              # (B, T)

            # Teacher forcing targets
            inp = captions[:, :-1]                        # (B, T-1)
            tgt = captions[:, 1:]                         # (B, T-1)

            logits, _ = self.forward(images, inp)         # (B, T-1, V)
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())
        return last_loss

    # -------- Forward --------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        # Encode images to a single feature vector per image
        feat = self.enc_proj(self.encoder(images))        # (B, hidden)
        h0 = feat.unsqueeze(0)                            # (1, B, hidden)
        c0 = torch.zeros_like(h0)                         # (1, B, hidden)
        if hidden_state is None:
            hidden_state = (h0, c0)

        if captions is not None:
            # Teacher forcing path
            emb = self.embedding(captions)                # (B, T, hidden)
            out, hidden_state = self.lstm(emb, hidden_state)  # (B, T, hidden)
            logits = self.fc_out(out)                     # (B, T, V)
            return logits, hidden_state

        # Greedy generation path
        B = images.size(0)
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)  # start with <SOS>
        logits_steps = []
        hidden = hidden_state

        for _ in range(self.max_len):
            emb = self.embedding(tokens[:, -1:])          # (B, 1, hidden)
            out, hidden = self.lstm(emb, hidden)          # (B, 1, hidden)
            step_logits = self.fc_out(out)                # (B, 1, V)
            logits_steps.append(step_logits)
            next_tok = step_logits.argmax(-1)             # (B, 1)
            tokens = torch.cat([tokens, next_tok], dim=1)

        logits = torch.cat(logits_steps, dim=1)           # (B, L, V)
        return logits, hidden
