import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple


# ---------- helpers ----------
def _first_int(x: Any) -> int:
    """Safely extract an int (e.g., vocab size) from possibly nested tuples/lists."""
    if isinstance(x, int):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _first_int(x[0])
    try:
        return int(x)
    except Exception:
        return 10000  # safe fallback


def _infer_in_channels(in_shape: Any) -> int:
    """Infer C from in_shape (C,H,W) with safe fallbacks."""
    if isinstance(in_shape, (tuple, list)):
        if len(in_shape) >= 1 and isinstance(in_shape[0], (int,)):
            return int(in_shape[0])
        if len(in_shape) >= 2 and isinstance(in_shape[1], (int,)):
            # some older templates used in_shape[1]
            return int(in_shape[1])
    return 3


# ---------- modules ----------
class ViTStyleEncoder(nn.Module):
    """
    CNN patch stem (conv with stride=patch_size) → tokens [B, P, H].
    """
    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Patchify with a conv "stem"
        self.stem = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

        # LayerNorm over the hidden_dim (last dim of token sequence)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] (e.g., 224x224 expected by your transforms)
        x = self.stem(x)                                # [B, Hdim, H', W']
        x = x.flatten(2).transpose(1, 2)                # [B, P, Hdim]
        x = self.norm(x)                                # [B, P, Hdim]
        return x


class TransformerDecoder(nn.Module):
    """
    Plain Transformer decoder: embeddings → TransformerDecoder → vocab projection.
    Expects memory from encoder as [B, P, H].
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, vocab_size: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Provided for API compatibility with LSTM-style decoders
        h0 = torch.zeros(batch, self.hidden_dim, device=device)
        c0 = torch.zeros(batch, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(
        self,
        inputs: torch.Tensor,                           # [B, T]
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        features: Optional[torch.Tensor] = None         # [B, P, H]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.dropout(self.embedding(inputs))      # [B, T, H]

        if hidden_state is None:
            batch = inputs.size(0)
            device = inputs.device
            hidden_state = self.init_zero_hidden(batch, device)

        if features is None:
            # If no memory is provided, make a trivial one from h0
            mem = hidden_state[0].unsqueeze(1)          # [B, 1, H]
        else:
            mem = features                               # [B, P, H]

        # Causal mask for autoregressive decoding
        T = emb.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=emb.device), diagonal=1).bool()

        out = self.decoder(tgt=emb, memory=mem, tgt_mask=tgt_mask)  # [B, T, H]
        logits = self.fc(out)                                       # [B, T, V]
        return logits, hidden_state


# ---------- top-level Net ----------
class Net(nn.Module):
    """
    ViT-style patch encoder → Transformer decoder for image captioning.
    Includes training hooks required by the harness.
    """
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device, *_, **__):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        self.in_channels = _infer_in_channels(in_shape)
        self.vocab_size = _first_int(out_shape)

        # hyperparams snapshot
        self.prm = dict(prm) if prm is not None else {}
        drop_p = float(self.prm.get("dropout", 0.2))

        # Encoder: conv-patchify to hidden_dim tokens
        hidden_dim = 640
        self.encoder = ViTStyleEncoder(in_channels=self.in_channels, patch_size=16, hidden_dim=hidden_dim)

        # Decoder: transformer
        self.decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=1,
            vocab_size=self.vocab_size,
            dropout=drop_p,
        )

        # training objects (bound in train_setup)
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # Instance method (some harnesses call this)
    def supported_hyperparameters(self):
        return {"lr", "momentum", "dropout"}

    def train_setup(self, prm: dict) -> None:
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        dropout = float(prm.get("dropout", self.prm.get("dropout", 0.2)))

        # update decoder dropout in case it's overridden
        self.decoder.dropout.p = dropout
        for m in self.decoder.decoder.layers:
            m.dropout.p = dropout  # ensure consistency

        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # Use momentum as AdamW beta1 so it's actually *used*
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data) -> None:
        """
        Robust training loop that iterates DataRoll without indexing.
        Expects batches like (images, captions, *extras) or dicts with 'x'/'y'.
        """
        prm = getattr(train_data, "prm", self.prm)
        self.train_setup(prm)

        for batch in train_data:
            # unpack a batch
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue
                images, captions = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("x", None); captions = batch.get("y", None)
                if images is None or captions is None:
                    continue
            else:
                images = getattr(batch, "x", None); captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue

            images = images.to(self.device)
            captions = captions.long().to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]  # handle [B,1,T]

            if captions.size(1) <= 1:
                continue

            # teacher forcing
            inputs = captions[:, :-1]   # remove EOS
            targets = captions[:, 1:]   # predict next

            features = self.encoder(images)                     # [B, P, H]
            logits, _ = self.decoder(inputs, None, features)    # [B, T-1, V]

            loss = self.criterion(logits.reshape(-1, self.vocab_size),
                                  targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

    def init_zero_hidden(self, batch: int, device: torch.device):
        # Provide a hidden-state initializer for API parity
        return self.decoder.init_zero_hidden(batch, device)

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Training:  returns (logits[B,T-1,V], hidden)
        Inference: returns (logits[B,1,V] for one BOS step, hidden)
        """
        images = images.to(self.device)
        features = self.encoder(images)                         # [B, P, H]

        if captions is not None:
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            captions = captions.long().to(self.device)
            if captions.size(1) <= 1:
                # nothing to predict; return a single dummy step
                bos = torch.full((images.size(0), 1), 1, dtype=torch.long, device=self.device)
                logits, hidden_state = self.decoder(bos, hidden_state, features)
                return logits, hidden_state

            inputs = captions[:, :-1]                           # [B, T-1]
            logits, hidden_state = self.decoder(inputs, hidden_state, features)  # [B, T-1, V]
            return logits, hidden_state

        # inference single-step with BOS=1
        bos = torch.full((images.size(0), 1), 1, dtype=torch.long, device=self.device)
        logits, hidden_state = self.decoder(bos, hidden_state, features)         # [B, 1, V]
        return logits.squeeze(1), hidden_state


# Module-level alias (some harnesses look for this function)
def supported_hyperparameters():
    return {"lr", "momentum", "dropout"}


# Factory expected by some trainers
def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
