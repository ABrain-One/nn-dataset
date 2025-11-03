import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm: Dict[str, Any], device: torch.device, *_, **__):
        super().__init__()
        # ---- API aliases ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm if prm is not None else {}
        self.device = device

        # Channels / vocab
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        try:
            # Expect out_shape like ((V,),) or (V,) or V
            self.vocab_size = int(out_shape[0][0])
        except Exception:
            try:
                self.vocab_size = int(out_shape[0])
            except Exception:
                self.vocab_size = int(out_shape)

        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Hyper-params
        self.d_model = int(self.prm.get('d_model', 768))
        self.nhead = int(self.prm.get('nhead', 8))
        self.num_layers = int(self.prm.get('num_layers', 1))
        self.max_len = int(self.prm.get('max_len', 20))
        self.sos_idx = int(self.prm.get('sos_idx', 1))

        # ----- Encoder (CNN -> global pooling -> projection) -----
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, self.d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.d_model),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, D, 1, 1]
            nn.Flatten(),                   # -> [B, D]
        )

        # ----- Decoder (Embedding + PosEnc + TransformerDecoder) -----
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=float(self.prm.get('dropout', 0.1)))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_layers)
        self.fc_final = nn.Linear(self.d_model, self.vocab_size)

        # Training bits (initialized in train_setup)
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    def train_setup(self, prm: Dict[str, Any]) -> None:
        """Initialize loss/optimizer with provided hyperparameters."""
        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)

        lr = float(prm.get('lr', 1e-3))
        momentum = float(prm.get('momentum', 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data) -> None:
        """Optional training loop placeholder (kept minimal to satisfy API)."""
        self.train()
        if self.optimizer is None or self.criterion is None:
            return  # train_setup not called; nothing to do
        for i, (images, captions) in enumerate(train_data):
            images = images.to(self.device)          # [B, C, H, W]
            captions = captions.to(self.device)      # [B, T]
            if captions.size(1) < 2:
                continue  # need at least <sos> and one target

            # Teacher forcing: predict t+1 from tokens up to t
            inputs = captions[:, :-1]                # [B, T-1]
            targets = captions[:, 1:]                # [B, T-1]

            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self.forward(images, inputs)  # [B, T-1, V]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            if i % 300 == 0:
                print(f"Batch {i}: Loss {float(loss.item()):.4f}")

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a single memory token per image: [B, 1, D]."""
        feats = self.encoder(images)             # [B, D]
        memory = feats.unsqueeze(1)              # [B, 1, D]
        return memory

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # True where positions should be masked
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        When captions is provided (teacher forcing):
            returns (logits [B, T, V], hidden_state_stub)
        Otherwise (inference-greedy):
            returns (logits [B, T, V], hidden_state_stub)
        """
        memory = self._encode(images)  # [B, 1, D]

        if captions is not None:
            if captions.dtype != torch.long:
                captions = captions.long()
            tgt = self.embedding(captions)                 # [B, T, D]
            tgt = self.pos_encoder(tgt)                    # [B, T, D]
            T = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, images.device)  # [T, T]
            decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)   # [B, T, D]
            logits = self.fc_final(decoded)                # [B, T, V]
            h_stub = memory.transpose(0, 1)                # [1, B, D]
            c_stub = torch.zeros_like(h_stub)
            return logits, (h_stub, c_stub)

        # Inference (greedy decode)
        B = images.size(0)
        ys = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)  # [B, 1]
        logits_steps = []
        for t in range(self.max_len):
            tgt = self.embedding(ys)                       # [B, t+1, D]
            tgt = self.pos_encoder(tgt)                    # [B, t+1, D]
            T = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, images.device)  # [T, T]
            decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)   # [B, T, D]
            step_logits = self.fc_final(decoded[:, -1:, :])                     # [B, 1, V]
            logits_steps.append(step_logits)
            next_token = step_logits.argmax(dim=-1)         # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)         # grow sequence

        logits = torch.cat(logits_steps, dim=1)             # [B, max_len, V]
        h_stub = memory.transpose(0, 1)                     # [1, B, D]
        c_stub = torch.zeros_like(h_stub)
        return logits, (h_stub, c_stub)
