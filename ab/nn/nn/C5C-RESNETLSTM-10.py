from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------------- Encoder ----------------
class SpatialEncoder(nn.Module):
    # Lightweight CNN that maps an image to a sequence of feature tokens [B, S, H]
    def __init__(self, in_channels: int = 3, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        c = [64, 192, 384, hidden_dim]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(c[0], c[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(c[1], c[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> [B, S, H] where S = h*w tokens
        x = self.cnn(x)
        b, h, ph, pw = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, ph * pw, h)
        return x


# ---------------- Decoder ----------------
class TransformerDecoder(nn.Module):
    # Transformer decoder that consumes target tokens and cross-attends to image tokens
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=min(3072, hidden_size * 4),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, memory: Tensor, captions: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # memory: [B, S, H], captions: [B, T]
        emb = self.drop(self.embedding(captions))  # [B, T, H]
        tgt_mask = self._causal_mask(emb.size(1), emb.device)
        logits = self.decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask)  # [B, T, H]
        logits = self.out(logits)  # [B, T, V]
        return logits, None

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return z, z


# ---------------- Net wrapper ----------------
class Net(nn.Module):
    # Minimal image-captioning style network with expected harness API
    def __init__(self, in_shape: Any, out_shape: Any, prm: Dict[str, float], device: torch.device, *_, **__) -> None:
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_heads = int(self.prm.get("num_heads", 8))
        self.num_layers = int(self.prm.get("num_layers", 3))
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 16))

        if self.hidden_size % self.num_heads != 0:
            self.hidden_size = max(self.num_heads, ((self.hidden_size // self.num_heads) + 1) * self.num_heads)

        self.encoder = SpatialEncoder(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_size,
            dropout=float(self.prm.get("dropout", 0.1)),
        )
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=float(self.prm.get("decoder_dropout", 0.1)),
            pad_idx=self.pad_idx,
        )

        # Aliases some harness utilities might look for
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
        # Handles (C,H,W) or (N,C,H,W); default 3
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) == 3 and all(isinstance(v, int) for v in in_shape):
                return int(in_shape[0])          # (C,H,W)
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])          # (N,C,H,W)
        return 3

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        return int(x)

    def _normalize_captions(self, captions: Tensor) -> Tensor:
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3:
            captions = captions[:, 0, :]
        return captions

    # ---- training helpers ----
    def train_setup(self, prm: Dict[str, float]):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        lr = float(prm.get("lr", self.prm.get("lr", 1e-3)))
        beta1 = float(prm.get("momentum", self.prm.get("momentum", 0.9)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
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

            logits, _ = self.forward(images, inputs)   # teacher-forcing path
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # ---- forward / inference ----
    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        hidden_state: Optional[Any] = None,
    ):
        """
        Training: captions given -> (logits [B,T,V], hidden_state)
        Eval (BLEU): captions None -> token_ids [B,<=max_len] (Tensor ONLY)
        """
        memory = self.encoder(images.to(self.device))  # [B, S, H]

        # Teacher forcing for training
        if captions is not None:
            captions = self._normalize_captions(captions.to(self.device).long())
            logits, _ = self.decoder(memory, captions)  # [B, T, V]
            return logits, hidden_state

        # Greedy decode for BLEU / inference
        B = images.size(0)
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)

        for _ in range(self.max_len - 1):
            logits, _ = self.decoder(memory, tokens)   # [B, T, V]
            step_logits = logits[:, -1, :]             # [B, V]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)  # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        return tokens  # <- IMPORTANT: Tensor, not tuple

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        # Just call forward(images) to share the same behavior
        self.eval()
        return self.forward(images)

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return z, z


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
