import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 672):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
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
        x = self.backbone(images)      # [B,H,1,1]
        x = x.view(x.size(0), 1, -1)   # [B,1,H]
        return x


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
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

        self.hidden_dim = int(prm.get("hidden_dim", 672))

        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=self.hidden_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=min(8, self.hidden_dim // 64),
                batch_first=True,
            ),
            num_layers=4,
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm["lr"],
            betas=(prm.get("momentum", 0.9), 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        steps = 0
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.encoder(images)  # [B,1,H]
            tgt_mask = self._causal_mask(inp.size(1), images.device)

            x = self.embedding(inp)  # [B,T-1,H]
            x = self.pos_encoding(x)  # [B,T-1,H]

            out = self.decoder(x, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out)

            loss = self.criteria[0](
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
        hidden_state: Optional[torch.Tensor] = None,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]
            tgt_mask = self._causal_mask(inp.size(1), images.device)

            x = self.embedding(inp)
            x = self.pos_encoding(x)
            out = self.decoder(x, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out)
            return logits, hidden_state

        batch_size = images.size(0)
        max_len = 20
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        for _ in range(max_len - 1):
            tgt_mask = self._causal_mask(generated.size(1), images.device)
            x = self.embedding(generated)
            x = self.pos_encoding(x)
            out = self.decoder(x, memory, tgt_mask=tgt_mask)
            step_logits = self.fc_out(out[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, None


def supported_hyperparameters():
    return {"lr", "momentum"}
