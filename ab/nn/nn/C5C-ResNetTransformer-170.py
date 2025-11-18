import torch
import torch.nn as nn
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        captions: torch.Tensor,
        memory: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(captions)
        context = memory.mean(dim=1, keepdim=True)
        context = context.expand(-1, emb.size(1), -1)
        x = torch.cat([emb, context], dim=-1)
        x = self.dropout(x)
        outputs, hidden = self.gru(x, hidden)
        logits = self.fc_out(outputs)
        return logits, hidden


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) else int(in_shape)
        if isinstance(out_shape, (tuple, list)):
            vocab_size = int(out_shape[0])
        else:
            vocab_size = int(out_shape)
        self.vocab_size = vocab_size
        self.hidden_dim = 768

        self.encoder = CNNEncoder(in_channels, hidden_dim=self.hidden_dim)
        self.decoder = GRUDecoder(vocab_size=self.vocab_size, hidden_dim=self.hidden_dim)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm.get('momentum', 0.9), 0.999)
        )

    def _normalize_captions(self, captions: torch.Tensor) -> torch.Tensor:
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
            captions = self._normalize_captions(captions)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)
            logits, _ = self.decoder(inputs, memory, None)

            loss = self.criteria[0](
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
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
        hidden_state: Optional[torch.Tensor] = None
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._normalize_captions(captions)
            inputs = captions[:, :-1]
            logits, hidden = self.decoder(inputs, memory, hidden_state)
            return logits, hidden
        else:
            batch_size = images.size(0)
            max_len = 20
            sos_idx = 1
            generated = torch.full(
                (batch_size, 1),
                sos_idx,
                dtype=torch.long,
                device=self.device
            )
            hidden = hidden_state
            for _ in range(max_len - 1):
                logits_step, hidden = self.decoder(generated[:, -1:], memory, hidden)
                next_token = logits_step[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            return generated, hidden
