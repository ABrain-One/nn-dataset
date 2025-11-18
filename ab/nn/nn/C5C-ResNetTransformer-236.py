import math
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn


def supported_hyperparameters():
    return {"lr", "momentum"}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, embed_dim]
        return self.net(x)


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Tuple,
        out_shape: Tuple,
        prm: dict,
        device: torch.device,
        *_,
        **__,
    ):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[1] if len(in_shape) > 1 else in_shape[0])
        else:
            in_channels = int(in_shape)

        self.hidden_dim = 768

        self.encoder = CNNEncoder(in_channels, self.hidden_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=float(prm.get("weight_decay", 1e-4)),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data: Iterable):
        assert self.optimizer is not None
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory_vec = self.encoder(images)  # [B, H]
            memory = memory_vec.unsqueeze(1)   # [B, 1, H]

            embedded = self.embedding(inputs)
            embedded = self.pos_encoding(embedded)

            seq_len = embedded.size(1)
            tgt_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=self.device),
                diagonal=1,
            )
            output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)

            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        max_length: int = 20,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory_vec = self.encoder(images)
        memory = memory_vec.unsqueeze(1)  # [B, 1, H]
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]

            embedded = self.embedding(inputs)
            embedded = self.pos_encoding(embedded)

            seq_len = embedded.size(1)
            tgt_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=self.device),
                diagonal=1,
            )
            output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            hidden_state = output[:, -1, :].unsqueeze(0)
            return logits, hidden_state

        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )

        for _ in range(max_length - 1):
            embedded = self.embedding(generated)
            embedded = self.pos_encoding(embedded)
            output = self.transformer_decoder(embedded, memory)
            step_logits = self.fc_out(output[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state
