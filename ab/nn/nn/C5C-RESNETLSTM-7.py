import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


# Module-level hook kept for your harness
def supported_hyperparameters():
    return {"lr", "momentum"}


class _ImageEncoder(nn.Module):
    """Small CNN -> global pooled feature -> linear to hidden size."""
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x).flatten(1)         # [B, 256]
        h0 = torch.tanh(self.proj(feat))      # [B, H]
        return h0.unsqueeze(0)                # [1, B, H]


class _GRUDecoder(nn.Module):
    """Token embedding -> GRU -> vocab projection. Hidden is init from image features."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        token_inputs: torch.Tensor,          # [B] or [B, T]
        hidden: Optional[torch.Tensor],      # [1, B, H] or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_inputs.dim() == 1:
            token_inputs = token_inputs.unsqueeze(1)  # [B, 1]
        emb = self.embedding(token_inputs)            # [B, T, E]
        out, hidden = self.gru(emb, hidden)          # out: [B, T, H]
        logits = self.fc(out)                        # [B, T, V]
        return logits, hidden


class Net(nn.Module):
    """
    Image -> CNN encoder -> init GRU hidden -> token GRU decoder -> logits over vocab.
    Keeps the same training/forward API.
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}
        self.device = device

        # Robustly infer channels and vocab size from possibly nested shapes
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Token indices (override via prm if needed)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))

        # Model dims (override via prm if needed) - hidden >= 640 per spec
        embed_dim = int(self.prm.get("embed_dim", 256))
        hidden_dim = int(self.prm.get("hidden_dim", 640))

        # Encoder & Decoder
        self.encoder = _ImageEncoder(self.in_channels, hidden_dim)
        self.decoder = _GRUDecoder(self.vocab_size, embed_dim, hidden_dim, pad_idx=self.pad_idx)

        # Backward-compat attribute aliases some harnesses expect
        self.cnn = self.encoder.cnn           # encoder backbone
        self.rnn = self.decoder.gru           # recurrent core
        self.embedding = self.decoder.embedding
        self.fc_out = self.decoder.fc

        # Training objects (set in train_setup)
        self.criteria: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # Some harnesses call this as a class/static method
    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _first_int(x: Any) -> int:
        """Extract an int (e.g., vocab size) from nested tuples/lists/ints."""
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        try:
            return int(x)
        except Exception:
            return 10000  # safe fallback

    @staticmethod
    def _infer_in_channels(in_shape: Any) -> int:
        """Infer channels from in_shape; handles (C,H,W) or (N,C,H,W)."""
        if isinstance(in_shape, (tuple, list)):
            # (C, H, W)
            if len(in_shape) == 3 and all(isinstance(v, int) for v in in_shape):
                return int(in_shape[0])
            # (N, C, H, W)
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])
        return 3

    # ---- Training API -----------------------------------------------------------
    def train_setup(self, prm):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criteria = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        """
        Minimal training loop stub that runs a teacher-forced step per batch.
        """
        self.train_setup(getattr(train_data, "prm", self.prm))
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
            if captions.size(1) <= 1:
                continue

            # Teacher forcing
            sos = torch.full((images.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos, captions[:, :-1]], dim=1)   # [B, T]
            targets = captions                                   # [B, T]

            # Encode image -> init hidden
            hidden0 = self.encoder(images)                        # [1, B, H]
            logits, _ = self.decoder(inputs, hidden0)             # [B, T, V]

            loss = self.criteria(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

    # ---- Inference/Forward ------------------------------------------------------
    def init_zero_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero hidden state (rarely used since we init from image)."""
        hidden_dim = self.decoder.gru.hidden_size
        return torch.zeros(1, batch_size, hidden_dim, device=device)

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If teacher_forcing and captions provided:
            returns (logits [B, T, V], hidden [1, B, H])
        Else:
            returns one-step from SOS: (logits [B, 1, V], hidden [1, B, H])
        """
        images = images.to(self.device)

        # Prepare captions if present
        if captions is not None and captions.ndim == 3:
            captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
        if captions is not None:
            captions = captions.to(self.device).long()

        # Hidden init: from image encoder if not given
        if hidden_state is None:
            hidden_state = self.encoder(images)  # [1, B, H]

        if teacher_forcing:
            assert captions is not None, "captions must be provided when teacher_forcing=True"
            sos = torch.full((images.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos, captions[:, :-1]], dim=1)   # [B, T]
            logits, hidden_state = self.decoder(inputs, hidden_state)  # [B, T, V]
            return logits, hidden_state
        else:
            # Single-step decode from SOS
            sos = torch.full((images.size(0),), self.sos_idx, dtype=torch.long, device=self.device)  # [B]
            logits, hidden_state = self.decoder(sos, hidden_state)  # [B, 1, V]
            return logits, hidden_state
