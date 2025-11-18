import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 640):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)            # [B,H,1,1]
        x = x.view(x.size(0), 1, -1)    # [B,1,H]
        return x


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        # in_shape ~ (C,H,W) or similar – only need channels
        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[0])
        else:
            in_channels = int(in_shape)

        # out_shape ~ (vocab_size,) or int
        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = 640
        self.num_layers = 2

        # Encoder
        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=self.hidden_dim)

        # Decoder
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim * 2,  # token + global image context
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm["lr"],
            weight_decay=prm.get("weight_decay", 0.01),
        )

    def _norm_caps(self, captions: torch.Tensor) -> torch.Tensor:
        # Accept [B,T] or [B,1,T] or [B,T,1]
        if captions.ndim == 3:
            if captions.size(1) == 1:
                captions = captions[:, 0, :]
            else:
                captions = captions[:, :, 0]
        return captions.long()

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)  # [B,T]

            # Teacher forcing: predict t+1 from 0..t
            inp = captions[:, :-1]  # [B,T-1]
            tgt = captions[:, 1:]   # [B,T-1]

            memory = self.encoder(images)  # [B,1,H]
            context = memory.mean(dim=1, keepdim=True)  # [B,1,H]

            emb = self.embedding(inp)  # [B,T-1,H]
            emb = self.pos_encoding(emb)
            emb = self.dropout(emb)

            ctx = context.expand(-1, emb.size(1), -1)  # [B,T-1,H]
            dec_inp = torch.cat([emb, ctx], dim=-1)    # [B,T-1,2H]

            outputs, _ = self.lstm(dec_inp)            # [B,T-1,H]
            logits = self.fc_out(outputs)              # [B,T-1,V]

            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                tgt.reshape(-1),
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
        hidden_state: Optional[tuple] = None,
        max_length: int = 20,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B,1,H]
        context = memory.mean(dim=1, keepdim=True)  # [B,1,H]

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]  # [B,T-1]

            emb = self.embedding(inp)
            emb = self.pos_encoding(emb)
            emb = self.dropout(emb)

            ctx = context.expand(-1, emb.size(1), -1)
            dec_inp = torch.cat([emb, ctx], dim=-1)

            outputs, hidden_state = self.lstm(dec_inp, hidden_state)
            logits = self.fc_out(outputs)
            return logits, hidden_state

        # Inference: greedy decode
        batch_size = images.size(0)
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            emb = self.embedding(generated[:, -1:])
            emb = self.pos_encoding(emb)
            emb = self.dropout(emb)

            ctx = context
            dec_inp = torch.cat([emb, ctx], dim=-1)
            outputs, hidden_state = self.lstm(dec_inp, hidden_state)
            step_logits = self.fc_out(outputs[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
