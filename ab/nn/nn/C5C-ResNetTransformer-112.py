import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Tuple[int, ...],
        prm: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 640

        in_channels = int(in_shape[0]) if len(in_shape) == 3 else int(in_shape[1])

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.enc_proj = nn.Linear(128, self.hidden_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        self.rnn = nn.GRU(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros(1, batch, self.hidden_dim, device=device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )
        self.criterion = self.criterion.to(self.device)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = self.enc_proj(x.flatten(1))
        return x.unsqueeze(1)  # [B, 1, H]

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data) -> float:
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

            memory = self._encode(images)
            ctx = memory.mean(dim=1, keepdim=True)

            emb = self.embedding(inputs)
            ctx_exp = ctx.expand(-1, emb.size(1), -1)
            rnn_in = torch.cat([emb, ctx_exp], dim=-1)
            rnn_in = self.dropout(rnn_in)

            h0 = self.init_zero_hidden(images.size(0), self.device)
            out, _ = self.rnn(rnn_in, h0)
            out = self.dropout(out)
            logits = self.fc_out(out)

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
        hidden_state=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, dtype=torch.float32)
        memory = self._encode(images)
        ctx = memory.mean(dim=1, keepdim=True)
        batch = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            inputs = captions[:, :-1]

            emb = self.embedding(inputs)
            ctx_exp = ctx.expand(-1, emb.size(1), -1)
            rnn_in = torch.cat([emb, ctx_exp], dim=-1)

            if hidden_state is None:
                hidden_state = self.init_zero_hidden(batch, self.device)

            out, hidden_state = self.rnn(rnn_in, hidden_state)
            logits = self.fc_out(out)
            return logits, hidden_state

        raise NotImplementedError("Inference without captions is not implemented")
