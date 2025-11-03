# -*- coding: utf-8 -*-
"""
Self-contained image captioning model with:
- Simple CNN encoder (no torchvision)
- LSTM decoder with teacher forcing during training and greedy decode at eval
- Exact public API names: class Net, and methods __init__, train_setup, learn, forward
- Module-level supported_hyperparameters()

Assumptions / conventions:
- PAD token id = 0, SOS token id = 1, EOS token id = 2
- forward() works for both train/infer:
    - Training: forward(images, captions, lengths, mode='train')
      returns logits of shape [B, T, vocab]
    - Inference: forward(images, mode='test', max_len=...)  (captions optional)
      returns greedy-decoded token ids [B, max_len]
- learn() accepts either a single batch tuple OR an iterable DataLoader of such batches.
  A "batch" is (images, captions, lengths) with:
      images: float tensor [B, C, H, W]
      captions: long tensor [B, T] (token ids; includes SOS at pos 0; PAD=0)
      lengths: long/int list/tensor with true lengths (including SOS, excluding any trailing PAD)
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small CNN encoder (no torchvision)
# -------------------------
class _EncoderCNN(nn.Module):
    """Simple conv encoder -> global avg pool -> linear projection to embed_dim."""
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        # A light ResNet-ish stack without residuals (keeps it dependency-free)
        self.features = nn.Sequential(
            # 1/2
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 1/4

            # 1/8
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            # 1/16
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            # 1/32 (optional extra depth)
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)             # [B, 512, H/32, W/32]
        x = self.avg(x).squeeze(-1).squeeze(-1)  # [B, 512]
        x = self.proj(x)                 # [B, embed_dim]
        return x


# -------------------------
# LSTM decoder
# -------------------------
class _DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        self.init_h = nn.Linear(embed_dim, hidden_size)
        self.init_c = nn.Linear(embed_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward_step(
        self, xt: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one time-step: xt [B, E] -> logits [B, V]."""
        xt = self.dropout(xt).unsqueeze(1)  # [B, 1, E]
        o, hidden = self.lstm(xt, hidden)   # o: [B, 1, H]
        logits = self.out(o.squeeze(1))     # [B, V]
        return logits, hidden

    def init_hidden(self, enc_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use image features to initialize hidden and cell states."""
        h0 = torch.tanh(self.init_h(enc_feats))
        c0 = torch.tanh(self.init_c(enc_feats))
        # shape [num_layers, B, H]
        h0 = h0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        return (h0, c0)

    def forward_train(
        self,
        enc_feats: torch.Tensor,      # [B, E]
        captions: torch.Tensor,       # [B, T] (includes SOS at t=0)
        lengths: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Teacher forcing: predict tokens 1..T-1 from inputs 0..T-2.
        Returns logits [B, T-1, V].
        """
        B, T = captions.shape
        device = captions.device

        hidden = self.init_hidden(enc_feats)

        # Inputs are tokens 0..T-2, targets are tokens 1..T-1
        inputs = captions[:, :-1]  # [B, T-1]
        targets_len = inputs.size(1)

        # Embed all at once, then unroll for clarity
        emb = self.embed(inputs)  # [B, T-1, E]

        logits_all = []
        for t in range(targets_len):
            xt = emb[:, t, :]                # [B, E]
            logits_t, hidden = self.forward_step(xt, hidden)  # [B, V]
            logits_all.append(logits_t.unsqueeze(1))

        logits = torch.cat(logits_all, dim=1)  # [B, T-1, V]
        return logits

    def forward_greedy(
        self,
        enc_feats: torch.Tensor,      # [B, E]
        sos_id: int,
        eos_id: int,
        max_len: int = 30,
    ) -> torch.Tensor:
        """Greedy decoding -> token ids [B, max_len]."""
        B = enc_feats.size(0)
        device = enc_feats.device

        hidden = self.init_hidden(enc_feats)
        prev = torch.full((B,), sos_id, dtype=torch.long, device=device)
        outputs = []

        for _ in range(max_len):
            xt = self.embed(prev)                   # [B, E]
            logits_t, hidden = self.forward_step(xt, hidden)  # [B, V]
            pred = torch.argmax(logits_t, dim=-1)   # [B]
            outputs.append(pred.unsqueeze(1))
            prev = pred

        return torch.cat(outputs, dim=1)            # [B, max_len]


# -------------------------
# Public model class with required API
# -------------------------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm: Dict[str, Any], device, *_, **__):
        """
        Args:
            in_shape: e.g., (B, C, H, W) or similar; we read C if available (fallback C=3)
            out_shape: vocab size (int) or tuple/list holding it
            prm: dict of hyperparameters
            device: torch.device or 'cuda'/'cpu'
        """
        super().__init__()

        # ---- API aliases (commonly auto-injected by caller) ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = torch.device(device)

        # Infer channels and vocab
        self.in_channels = (
            int(in_shape[1]) if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3
        )
        # out_shape could be int, or (vocab_size,) or nested — be defensive:
        if isinstance(out_shape, (tuple, list)):
            # pull first int-like we see
            flat = []
            for x in out_shape:
                if isinstance(x, (tuple, list)):
                    flat.extend(list(x))
                else:
                    flat.append(x)
            self.vocab_size = int(flat[0])
        else:
            self.vocab_size = int(out_shape)

        # ---- Special tokens and defaults ----
        self.PAD = int(prm.get("pad_id", 0))
        self.SOS = int(prm.get("sos_id", 1))
        self.EOS = int(prm.get("eos_id", 2))

        # Model dims
        self.embed_dim = int(prm.get("embed_dim", 512))
        self.hidden_size = int(prm.get("hidden_size", 512))
        self.num_layers = int(prm.get("num_layers", 1))
        self.dropout = float(prm.get("dropout", 0.1))
        self.max_len = int(prm.get("max_len", 30))

        # Modules
        self.encoder = _EncoderCNN(self.in_channels, self.embed_dim)
        self.decoder = _DecoderLSTM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pad_idx=self.PAD,
        )

        # Training helpers (set in train_setup)
        self.criteria: Tuple[nn.Module, ...] = tuple()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    # -------------------------
    # REQUIRED PUBLIC METHOD
    # -------------------------
    def train_setup(self, prm: Dict[str, Any]):
        """
        Initializes optimizer/criteria/scheduler. Keeps API names intact.
        Uses AdamW with 'lr' and 'momentum' (mapped to beta1).
        """
        self.to(self.device)
        self.train()

        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        weight_decay = float(prm.get("weight_decay", 0.0))

        # CrossEntropy over vocabulary, ignoring PAD
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.PAD).to(self.device),)

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
            weight_decay=weight_decay,
        )

        # Optional scheduler (ReduceLROnPlateau by default)
        if prm.get("use_scheduler", True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2, verbose=False
            )
        else:
            self.scheduler = None

    # -------------------------
    # REQUIRED PUBLIC METHOD
    # -------------------------
    def learn(
        self,
        train_data: Union[
            Tuple[torch.Tensor, torch.Tensor, Union[List[int], torch.Tensor]],
            Iterable[Tuple[torch.Tensor, torch.Tensor, Union[List[int], torch.Tensor]]],
        ],
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        Trains for ONE pass over the provided data.
        - If `train_data` is a single batch, optimizes on that batch.
        - If `train_data` is an iterable (e.g., DataLoader), loops through all batches once.

        Returns:
            loss_value (float), outputs (dict with 'logits' or 'preds' optionally)
        """
        assert self.optimizer is not None and len(self.criteria) > 0, \
            "Call train_setup(prm) before learn()."

        self.train()
        total_loss = 0.0
        n_batches = 0
        last_out: Dict[str, torch.Tensor] = {}

        def _train_on_batch(batch) -> float:
            images, captions, lengths = batch
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)
            if isinstance(lengths, torch.Tensor):
                lengths_ = lengths.to(self.device)
            else:
                lengths_ = torch.tensor(lengths, device=self.device, dtype=torch.long)

            # Forward (teacher forcing)
            logits = self.forward(images, captions, lengths_, mode="train")  # [B, T-1, V]

            # Targets are next tokens (captions[:, 1:])
            targets = captions[:, 1:]  # [B, T-1]
            B, Tm1, V = logits.shape
            loss = self.criteria[0](logits.reshape(B * Tm1, V), targets.reshape(B * Tm1))

            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                # Plateau schedulers typically want a validation loss; we use current batch loss
                self.scheduler.step(loss.detach())

            last_out.clear()
            last_out["logits"] = logits.detach()
            return float(loss.detach().item())

        # Single batch or iterable
        if isinstance(train_data, tuple) and len(train_data) == 3:
            total_loss = _train_on_batch(train_data)
            n_batches = 1
        else:
            for batch in train_data:
                total_loss += _train_on_batch(batch)
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, last_out

    # -------------------------
    # REQUIRED PUBLIC METHOD
    # -------------------------
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        lengths: Optional[Union[List[int], torch.Tensor]] = None,
        mode: str = "test",
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Universal forward:
          - mode='train' and (captions provided) -> returns logits [B, T-1, V]
          - mode!='train' -> greedy decode, returns token ids [B, L]
        """
        self.eval() if mode != "train" else self.train()

        images = images.to(self.device, non_blocking=True)
        enc_feats = self.encoder(images)  # [B, E]

        if mode == "train" and captions is not None:
            captions = captions.to(self.device, non_blocking=True)
            return self.decoder.forward_train(enc_feats, captions, lengths)

        # Inference (greedy)
        max_len = int(max_len or self.max_len)
        preds = self.decoder.forward_greedy(
            enc_feats, sos_id=self.SOS, eos_id=self.EOS, max_len=max_len
        )
        return preds


# -------------------------
# Module-level API
# -------------------------
def supported_hyperparameters():
    # Keep exactly the names you've been using in your pipeline
    return {"lr", "momentum", "dropout", "weight_decay", "embed_dim", "hidden_size", "num_layers", "max_len"}
