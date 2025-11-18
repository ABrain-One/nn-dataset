import math
import torch
import torch.nn as nn
from typing import Any, Iterable, Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,H]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=min(2048, d_model * 2),
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        inputs: torch.Tensor,           # [B,T]
        hidden_state: Optional[torch.Tensor],
        memory: torch.Tensor,           # [B,S,H]
    ):
        emb = self.embedding(inputs)    # [B,T,H]
        emb = self.pos_encoding(emb)
        T = inputs.size(1)
        tgt_mask = self._causal_mask(T, inputs.device)
        out = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)  # [B,T,H]
        logits = self.fc_out(out)       # [B,T,V]
        return logits, hidden_state


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 768):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x.unsqueeze(1)  # [B,1,out_channels]


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[0])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = prm.get("hidden_dim", 768)
        self.layers = prm.get("layers", 4)

        self.encoder = CNNEncoder(in_channels=in_channels, out_channels=self.hidden_dim)
        self.rnn = TransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.hidden_dim,
            num_layers=self.layers,
            nhead=min(8, self.layers * 2),
        )

        self.criterion = None
        self.optimizer = None

    def train_setup(self, prm: dict, **kwargs):
        self.to(self.device)
        lr = prm["lr"]
        momentum = prm.get("momentum", 0.9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )

    def _norm_caps(self, captions: torch.Tensor) -> torch.Tensor:
        # Expect [B,T] or [B,1,T]
        if captions.ndim == 3:
            if captions.size(1) == 1:
                captions = captions[:, 0, :]
            else:
                captions = captions[:, :, 0]
        return captions.long()

    def learn(self, train_data: Iterable):
        """
        train_data: iterable over (images, captions) pairs.
        """
        assert self.optimizer is not None, "Call train_setup(prm) before learn()"
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)  # [B,T]

            assert images.dim() == 4
            assert captions.ndim == 2

            inputs = captions[:, :-1]  # [B,T-1]
            targets = captions[:, 1:]  # [B,T-1]

            memory = self.encoder(images)  # [B,1,H]

            logits, hidden_state = self.rnn(inputs, None, memory)  # [B,T-1,V]

            assert logits.shape[0] == inputs.shape[0]
            assert logits.shape[1] == inputs.shape[1]
            assert logits.shape[-1] == self.vocab_size

            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
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
        memory = self.encoder(images)  # [B,1,H]
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state

        # Inference: greedy decode
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            logits, hidden_state = self.rnn(generated, hidden_state, memory)
            step_logits = logits[:, -1, :]  # [B,V]
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
