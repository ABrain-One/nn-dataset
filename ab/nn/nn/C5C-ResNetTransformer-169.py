import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 768,
                 num_layers: int = 3, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, captions: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.embedding(captions)
        x = self.pos_encoding(x)
        tgt_len = x.size(1)
        tgt_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float('-inf'), device=x.device),
            diagonal=1
        )
        x = self.decoder(x, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(x)
        return logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) else int(in_shape)
        if isinstance(out_shape, (tuple, list)):
            vocab_size = int(out_shape[0])
        else:
            vocab_size = int(out_shape)
        self.vocab_size = vocab_size
        self.hidden_dim = 768

        self.encoder = Encoder(in_channels, hidden_dim=self.hidden_dim)
        self.decoder = Decoder(vocab_size=self.vocab_size, hidden_dim=self.hidden_dim)

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
            logits = self.decoder(inputs, memory)

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
            logits = self.decoder(inputs, memory)
            return logits, None
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
            for _ in range(max_len - 1):
                logits_step = self.decoder(generated, memory)
                next_token = logits_step[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            return generated, None
