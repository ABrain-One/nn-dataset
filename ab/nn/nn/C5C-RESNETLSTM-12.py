import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- Shapes / sizes ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_layers = int(self.prm.get("num_layers", 1))
        self.pad_idx = int(self.prm.get("pad_idx", 0))   # also used as EOS by default
        self.sos_idx = int(self.prm.get("sos", self.prm.get("sos_idx", 1)))
        self.eos_idx = int(self.prm.get("eos", self.pad_idx))
        self.max_len = int(self.prm.get("max_length", self.prm.get("max_len", 50)))
        self.grad_clip_value = float(self.prm.get("grad_clip", 3.0))

        # ---- Modules ----
        self.encoder = NetEncoder(self.in_channels, self.hidden_size, device)
        self.decoder = NetDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            pad_idx=self.pad_idx,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
            dropout=float(self.prm.get("dropout", 0.1)),
            device=device,
        )

        # training helpers (populated in train_setup)
        self.criteria = None
        self.criterion = None
        self.optimizer = None

        self.to(self.device)

    # ---------------- API ----------------
    def train_setup(self, prm):
        prm = prm or {}
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        """Simple teacher-forcing training loop."""
        self.train()
        if self.optimizer is None:
            self.train_setup(self.prm)

        for batch in train_data:
            images, captions = batch
            images = images.to(self.device)
            captions = captions.to(self.device).long()

            # If captions are [B, 1, T] -> [B, T]
            if captions.dim() == 3 and captions.size(1) == 1:
                captions = captions[:, 0, :]

            # Need at least 2 tokens for input/target shift
            if captions.dim() != 2 or captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]   # feed to decoder
            targets = captions[:, 1:]   # predict next token

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_value)
            self.optimizer.step()

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Training: captions is [B, T_in]; returns (logits [B, T_in, V], hidden).
        Inference: captions is None; returns (generated_ids [B, <=max_len], hidden).
        """
        # Encode images -> feature vectors [B, H]
        features = self.encoder(images.to(self.device))

        if captions is not None:
            # Teacher forcing
            logits, hidden_state = self.decoder(
                images=None, captions=captions.to(self.device), features=features, hidden_state=hidden_state
            )
            return logits, hidden_state

        # Greedy decoding
        generated = self.decoder.decode(features, hidden_state=hidden_state, max_len=self.max_len)
        return generated, None

    # ---------------- helpers ----------------
    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        try:
            if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 and isinstance(in_shape[1], int):
                return in_shape[1]
        except Exception:
            pass
        return 3

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            for item in x:
                try:
                    return Net._first_int(item)
                except Exception:
                    continue
        return int(x)


class NetEncoder(nn.Module):
    """Tiny CNN encoder (torchvision-free) producing a feature vector [B, hidden_size]."""
    def __init__(self, in_channels: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gap(x).flatten(1)     # [B, 256]
        x = self.fc(x)                 # [B, H]
        return x


class NetDecoder(nn.Module):
    """GRU-based caption decoder with greedy decode."""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Map encoder features -> initial hidden state
        self.init_h_proj = nn.Linear(hidden_size, hidden_size)

    def _init_hidden_from_features(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, H] -> h0: [num_layers, B, H]
        h0 = torch.tanh(self.init_h_proj(features)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0.contiguous()

    def forward(
        self,
        images: Optional[torch.Tensor],
        captions: torch.Tensor,               # [B, T]
        features: torch.Tensor,               # [B, H]
        hidden_state: Optional[torch.Tensor] = None,  # [num_layers, B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = captions.size(0), captions.size(1)
        h0 = self._init_hidden_from_features(features) if hidden_state is None else hidden_state

        emb = self.embedding(captions)        # [B, T, H]
        out, h_n = self.gru(emb, h0)          # out: [B, T, H]
        logits = self.fc(out)                 # [B, T, V]
        return logits, h_n

    def decode(
        self,
        features: torch.Tensor,                       # [B, H]
        hidden_state: Optional[torch.Tensor] = None,  # [num_layers, B, H]
        max_len: int = 50,
    ) -> torch.Tensor:
        B = features.size(0)
        h = self._init_hidden_from_features(features) if hidden_state is None else hidden_state

        # Start tokens
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(tokens[:, -1:])  # [B, 1, H]
            out, h = self.gru(emb, h)            # [B, 1, H]
            logits = self.fc(out)                # [B, 1, V]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            generated.append(next_tok)
            tokens = torch.cat([tokens, next_tok], dim=1)

            # stop if all sequences hit EOS
            if (next_tok == self.eos_idx).all():
                break

        # Concatenate generated tokens after SOS (drop the initial SOS column)
        if generated:
            return tokens[:, 1:]
        return torch.empty(B, 0, dtype=torch.long, device=self.device)
