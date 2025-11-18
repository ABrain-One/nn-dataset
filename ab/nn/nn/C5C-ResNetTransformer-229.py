import torch
import torch.nn as nn
from typing import Any, Iterable, Optional, Tuple


def supported_hyperparameters():
    return {"lr", "momentum"}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)
        x = self.proj(x.flatten(1))
        return x.unsqueeze(1)


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Any,
        out_shape: Any,
        prm: dict,
        device: torch.device,
        *_,
        **__,
    ):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[1] if len(in_shape) > 1 else in_shape[0])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_size = 768
        self.encoder = CNNEncoder(in_channels, self.hidden_size)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
            ),
            num_layers=prm.get("num_layers", 1),
        )
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
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

            memory = self.encoder(images)  # [B,1,H]
            embedded = self.embedding(inputs)
            seq_len = embedded.size(1)
            tgt_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=self.device),
                diagonal=1,
            )
            output = self.decoder(embedded, memory=memory, tgt_mask=tgt_mask)
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
        memory = self.encoder(images)  # [B,1,H]
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]

            embedded = self.embedding(inputs)
            seq_len = embedded.size(1)
            tgt_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=self.device),
                diagonal=1,
            )
            output = self.decoder(embedded, memory=memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            hidden_state = output[:, -1, :].unsqueeze(0)
            return logits, hidden_state

        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            embedded = self.embedding(generated[:, -1:])
            output = self.decoder(embedded, memory=memory)
            step_logits = self.fc_out(output[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state
