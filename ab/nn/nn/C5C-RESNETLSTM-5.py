import torch
import torch.nn as nn
from typing import Optional, Tuple, Any


# ---------------------------------------------------------------------
# Public API hook kept for compatibility
# ---------------------------------------------------------------------
def supported_hyperparameters():
    return {"lr", "momentum"}


# ---------------------------------------------------------------------
# Simple image-conditioned RNN decoder
# ---------------------------------------------------------------------
class _ImageConditionedGRU(nn.Module):
    """
    A small CNN encodes the image -> vector.
    That vector initializes the GRU hidden state.
    Then we decode tokens (teacher-forced or one-step).
    """
    def __init__(
        self,
        vocab_size: int,
        in_channels: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        device: Optional[torch.device] = None,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.pad_idx = pad_idx

        # Image encoder → feature vector
        self.encoder = nn.Sequential(
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
        self.enc_to_hidden = nn.Linear(256, hidden_dim)

        # Token embedding + GRU
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True)

        # Projection to vocab
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images).flatten(1)             # [B, 256]
        h0 = torch.tanh(self.enc_to_hidden(feats))          # [B, H]
        return h0.unsqueeze(0)                              # [1, B, H] (GRU expected)

    def forward(
        self,
        inputs: torch.Tensor,          # [B] or [B, T]
        hidden: Optional[torch.Tensor],
        features: torch.Tensor,        # images: [B, C, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, T, V]  (if inputs is [B, T]) or [B, 1, V] (if inputs is [B])
            hidden: [1, B, H]
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)  # [B, 1]

        if hidden is None:
            hidden = self._encode_image(features)  # [1, B, H]

        emb = self.embedding(inputs)              # [B, T, E]
        out, hidden = self.gru(emb, hidden)       # out: [B, T, H]
        logits = self.fc_out(out)                 # [B, T, V]
        return logits, hidden


# ---------------------------------------------------------------------
# Main Net
# ---------------------------------------------------------------------
class Net(nn.Module):
    """
    Image → (CNN) → initial GRU hidden → token GRU decoder → vocab logits.
    """
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device, *_, **__):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}
        self.device = device

        # Infer channels and vocab size robustly
        self.in_channels = self._infer_in_channels(in_shape)
        self.vocab_size = self._first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Indices (PAD=0, SOS defaults to 1 if not provided)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))

        # Model dims
        embed_dim = int(self.prm.get("embed_dim", 256))
        hidden_dim = int(self.prm.get("hidden_dim", 512))

        # Decoder (with its own encoder inside)
        self.rnn = _ImageConditionedGRU(
            vocab_size=self.vocab_size,
            in_channels=self.in_channels,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=self.device,
            pad_idx=self.pad_idx,
        )

        # Training objects (initialized in train_setup)
        self.criteria = None
        self.optimizer = None

        self.to(self.device)

    # ---- Helpers ----------------------------------------------------------------
    @staticmethod
    def _first_int(x: Any) -> int:
        """Safely extract an int (e.g., vocab size) from possibly nested tuples/lists."""
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
        """Infer channel count from in_shape. Handles (C,H,W) or (N,C,H,W)."""
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 1 and isinstance(in_shape[0], int):
                return int(in_shape[0])  # (C,H,W)
            if len(in_shape) >= 2 and isinstance(in_shape[1], int):
                return int(in_shape[1])  # (N,C,H,W)
        return 3

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

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
        Minimal training loop stub (kept for compatibility with your harness).
        Iterates once through `train_data`, performs standard teacher-forced step.
        """
        self.train_setup(getattr(train_data, "prm", {}))
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device).long()
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
            if captions.size(1) <= 1:
                continue

            # Teacher forcing
            sos = torch.full((images.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos, captions[:, :-1]], dim=1)             # [B, T-1+1] = [B, T]
            targets = captions                                              # [B, T]

            logits, _ = self.rnn(inputs, None, features=images)             # [B, T, V]
            loss = self.criteria(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

    # ---- Forward ----------------------------------------------------------------
    def forward(self, images, teacher_forcing: bool = True, captions: Optional[torch.Tensor] = None):
        """
        If teacher_forcing and captions provided:
            returns logits: [B, T, V], hidden: [1, B, H]
        Else (inference):
            returns logits: [B, 1, V] (one step from SOS), hidden: [1, B, H]
        """
        images = images.to(self.device)

        if captions is not None and captions.ndim == 3:
            captions = captions[:, 0, :]  # [B,1,T] -> [B,T]
        if teacher_forcing:
            assert captions is not None, "captions must be provided when teacher_forcing=True"
            captions = captions.to(self.device).long()

            sos = torch.full((images.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos, captions[:, :-1]], dim=1)  # [B, T]
            outputs, hidden_state = self.rnn(inputs, None, features=images)  # [B, T, V], [1,B,H]
            return outputs, hidden_state
        else:
            # Single-step inference from SOS
            sos = torch.full((images.size(0),), self.sos_idx, dtype=torch.long, device=self.device)  # [B]
            outputs, hidden_state = self.rnn(sos, None, features=images)  # [B,1,V], [1,B,H]
            return outputs, hidden_state
