import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Iterable, Optional, Tuple


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)
        x = self.proj(x.flatten(1))
        return x.unsqueeze(1)  # [B,1,H]


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

        self.hidden_size = prm.get("hidden_size", 768)

        # Encoder
        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=self.hidden_size)

        # Decoder: GRU
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        self.criterion = None
        self.optimizer = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = prm["lr"]
        momentum = prm.get("momentum", 0.9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data: Iterable):
        assert self.optimizer is not None, "Call train_setup(prm) before learn()"
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.encoder(images)    # [B,1,H]
            ctx = memory.mean(dim=1, keepdim=True)

            emb = self.embedding(inp)
            ctx_expanded = ctx.expand(-1, emb.size(1), -1)
            dec_inp = torch.cat([emb, ctx_expanded], dim=-1)

            outputs, _ = self.gru(dec_inp)
            logits = self.fc_out(outputs)    # [B,T-1,V]

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
        hidden_state: Optional[torch.Tensor] = None,
        max_length: int = 20,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        ctx = memory.mean(dim=1, keepdim=True)  # [B,1,H]
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]

            emb = self.embedding(inp)
            ctx_expanded = ctx.expand(-1, emb.size(1), -1)
            dec_inp = torch.cat([emb, ctx_expanded], dim=-1)

            outputs, hidden_state = self.gru(dec_inp, hidden_state)
            logits = self.fc_out(outputs)
            return logits, hidden_state

        # Inference
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            emb = self.embedding(generated[:, -1:])
            ctx_expanded = ctx
            dec_inp = torch.cat([emb, ctx_expanded], dim=-1)
            outputs, hidden_state = self.gru(dec_inp, hidden_state)
            step_logits = self.fc_out(outputs[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
