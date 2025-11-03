import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _deep_first_int(x: Union[int, Tuple, List]) -> int:
    """Pull the first integer-like leaf out of a possibly nested out_shape."""
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _deep_first_int(x[0])
    return int(x)


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    """
    Encoder: small CNN → global pooled → Linear to hidden_size
    Decoder: GRU that takes [embed(token) ; image_features] at each step, then Linear → vocab
    """

    def __init__(self, in_shape, out_shape, prm: Dict[str, Any], device: torch.device, *_, **__):
        super().__init__()
        self.device = device

        # --- Shapes & sizes ---
        # in_shape may be (C, H, W) or (B, C, H, W). We follow prior code and read channels from index 1 when present.
        if isinstance(in_shape, (tuple, list)) and len(in_shape) >= 2:
            self.in_channels = int(in_shape[1])
        else:
            self.in_channels = 3

        self.vocab_size = _deep_first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # --- Hyperparameters with sane defaults ---
        self.hidden_size = int(prm.get('hidden_size', 768))
        self.num_layers = int(prm.get('num_layers', 1))
        self.dropout = float(prm.get('dropout', 0.3))
        self.pad_idx = int(prm.get('pad_idx', 0))
        self.sos_idx = int(prm.get('sos_idx', 1))
        self.eos_idx = int(prm.get('eos_idx', 2))
        self.max_len = int(prm.get('max_len', 30))

        # --- Encoder: simple, stable CNN backbone → 512 → hidden_size ---
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_size),
        )

        # --- Decoder: token embedding + GRU (consumes [embed ; img_feat]) + classifier ---
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.gru = nn.GRU(
            input_size=self.hidden_size * 2,   # [embed ; image_features]
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------- Training API ----------
    def train_setup(self, prm: Dict[str, Any]):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm['lr']),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Expects an iterable of (images, captions) where captions are [B,T] or [B,1,T].
        Teacher forcing is used: inputs=tokens[:, :-1], targets=tokens[:, 1:].
        """
        assert self.criteria and self.optimizer is not None, "Call train_setup(prm) before learn()."
        self.train()

        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            # teacher forcing split
            inputs = captions[:, :-1]   # [B, T-1]
            targets = captions[:, 1:]   # [B, T-1]

            # forward
            logits, _ = self.forward(images, captions=inputs)  # logits: [B, T-1, V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            # (optional) you can yield or log loss here

    # ---------- Inference helpers ----------
    def init_zero_hidden(self, batch_size: int) -> torch.Tensor:
        """GRU hidden state: [num_layers, B, hidden]"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

    # ---------- Forward ----------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If captions is provided (teacher forcing):
            Returns (logits [B, T, V], h_n [num_layers, B, H])
        Else (greedy generation):
            Returns (logits [B, Tgen, V], tokens [B, Tgen])  -- here second tensor is the generated ids
        """
        B = images.size(0)
        img_feats = self.encoder(images.to(self.device))                 # [B, H]
        img_feats_rep: Optional[torch.Tensor] = None

        if captions is not None:  # teacher forcing path
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            captions = captions.to(self.device)                           # [B, T]
            T = captions.size(1)
            if T == 0:
                empty_logits = torch.zeros(B, 0, self.vocab_size, device=self.device)
                return empty_logits, self.init_zero_hidden(B)

            # embed & concat image features per step
            emb = self.embedding(captions)                                # [B, T, H]
            if img_feats_rep is None:
                img_feats_rep = img_feats.unsqueeze(1).expand(B, T, -1)   # [B, T, H]
            gru_in = torch.cat([emb, img_feats_rep], dim=-1)              # [B, T, 2H]

            h0 = hidden_state if hidden_state is not None else self.init_zero_hidden(B)
            out, h_n = self.gru(gru_in, h0)                               # out: [B, T, H]
            logits = self.fc(out)                                         # [B, T, V]
            return logits, h_n

        # --- Greedy decoding ---
        T = self.max_len
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)  # [B, 1]
        h = hidden_state if hidden_state is not None else self.init_zero_hidden(B)
        logits_acc = []

        for _ in range(T):
            emb = self.embedding(tokens[:, -1:])                          # last step, shape [B,1,H]
            if img_feats_rep is None:
                img_feats_rep = img_feats.unsqueeze(1)                    # [B,1,H]
            step_in = torch.cat([emb, img_feats_rep], dim=-1)             # [B,1,2H]
            out, h = self.gru(step_in, h)                                 # out: [B,1,H]
            step_logits = self.fc(out)                                    # [B,1,V]
            logits_acc.append(step_logits)
            next_tok = step_logits.argmax(-1)                              # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)                 # grow
            if (next_tok.squeeze(1) == self.eos_idx).all():
                break

        if logits_acc:
            logits = torch.cat(logits_acc, dim=1)                         # [B, Tgen, V]
        else:
            logits = torch.zeros(B, 0, self.vocab_size, device=self.device)

        # drop the initial SOS from tokens for return
        return logits, tokens[:, 1:]
