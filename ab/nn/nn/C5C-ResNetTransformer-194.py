import torch
import torch.nn as nn
from typing import Optional, Tuple


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.proj(x)
        x = x.unsqueeze(1)
        return x  # [B, 1, H]


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        captions: torch.Tensor,
        memory: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(captions)
        context = memory.mean(dim=1, keepdim=True)
        context = context.expand(-1, emb.size(1), -1)
        x = torch.cat([emb, context], dim=-1)
        x = self.dropout(x)
        outputs, hidden = self.lstm(x, hidden)
        logits = self.fc_out(outputs)
        return logits, hidden


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[1])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            vocab_size = int(out_shape[0])
        else:
            vocab_size = int(out_shape)

        self.vocab_size = vocab_size
        self.hidden_dim = int(prm.get('hidden_dim', 768))

        self.features = CNNBackbone(in_channels, hidden_dim=self.hidden_dim)
        self.decoder = LSTMDecoder(vocab_size=self.vocab_size, hidden_dim=self.hidden_dim)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm.get('momentum', 0.9), 0.999),
        )

    def _normalize_captions(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data) -> float:
        self.train()
        total_loss = 0.0
        steps = 0
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._normalize_captions(captions)

            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.features(images)
            logits, _ = self.decoder(inp, memory, None)

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
        x: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        images = x.to(self.device, dtype=torch.float32)
        memory = self.features(images)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._normalize_captions(captions)
            inp = captions[:, :-1]
            logits, hidden = self.decoder(inp, memory, hidden_state)
            return logits, hidden

        batch_size = images.size(0)
        max_len = 20
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1),
            sos_idx,
            dtype=torch.long,
            device=self.device,
        )
        hidden = hidden_state
        for _ in range(max_len - 1):
            logits_step, hidden = self.decoder(generated[:, -1:], memory, hidden)
            next_tok = logits_step[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden
