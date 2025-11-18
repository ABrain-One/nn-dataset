import torch
import torch.nn as nn
from typing import Any, Iterable, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)
        x = self.proj(x.flatten(1))
        return x.unsqueeze(1)  # [B,1,H]


class Net(nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        out_shape: tuple,
        prm: dict,
        device: torch.device,
        *_,
        **__,
    ):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0]) if isinstance(out_shape, (tuple, list)) else int(out_shape)
        self.hidden_dim = 768

        # in_shape often (B,C,H,W) or (None,C,H,W)
        if len(in_shape) > 1:
            in_channels = int(in_shape[1])
        else:
            in_channels = int(in_shape[0])

        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=self.hidden_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.hidden_dim, self.vocab_size)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros(1, batch, self.hidden_dim, device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get("lr", 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get("momentum", 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(self.device)

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data: Iterable):
        """
        train_data: iterable of (images, captions)
        """
        assert self.optimizer is not None, "Call train_setup(prm) before learn()"
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions.to(self.device)
            caps = self._norm_caps(caps)  # [B,T]

            inputs = caps[:, :-1]  # [B,T-1]
            targets = caps[:, 1:]  # [B,T-1]

            memory = self.encoder(images)              # [B,1,H]
            context = memory.mean(dim=1, keepdim=True) # [B,1,H]

            emb = self.embedding(inputs)               # [B,T-1,H]
            ctx = context.expand(-1, emb.size(1), -1)
            dec_in = torch.cat([emb, ctx], dim=-1)     # [B,T-1,2H]

            outputs, _ = self.gru(dec_in)              # [B,T-1,H]
            logits = self.classifier(outputs)          # [B,T-1,V]

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
        memory = self.encoder(images)              # [B,1,H]
        context = memory.mean(dim=1, keepdim=True) # [B,1,H]
        batch_size = images.size(0)

        if captions is not None:
            caps = captions.to(self.device)
            caps = self._norm_caps(caps)
            inputs = caps[:, :-1]

            emb = self.embedding(inputs)
            ctx = context.expand(-1, emb.size(1), -1)
            dec_in = torch.cat([emb, ctx], dim=-1)

            outputs, hidden_state = self.gru(dec_in, hidden_state)
            logits = self.classifier(outputs)
            return logits, hidden_state

        # Inference (greedy)
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            emb = self.embedding(generated[:, -1:])
            ctx = context
            dec_in = torch.cat([emb, ctx], dim=-1)
            outputs, hidden_state = self.gru(dec_in, hidden_state)
            step_logits = self.classifier(outputs[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state
