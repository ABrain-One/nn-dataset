import torch
import torch.nn as nn
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNBody(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 768):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_channels = output_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.body(images)
        x = x.flatten(2).transpose(1, 2)
        return x


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        captions: torch.Tensor,
        memory: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
        self.hidden_dim = 768

        self.encoder = CNNBody(in_channels, output_channels=self.hidden_dim)
        self.decoder = LSTMDecoder(vocab_size=self.vocab_size, hidden_dim=self.hidden_dim)

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
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
