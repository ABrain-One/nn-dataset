import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

def supported_hyperparameters() -> set:
    return {'lr', 'momentum'}


class EncoderCNN(nn.Module):
    def __init__(self, input_channels: int = 3, output_dim: int = 768):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, output_dim, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.flatten(2).transpose(1, 2)
        return x  # [B, S, D]


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

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, input_dim: int = 768,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim

        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.pos_encoding = PositionalEncoding(input_dim, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, vocab_size)

    def forward(self, captions: Tensor, memory: Tensor) -> Tensor:
        x = self.embedding(captions)
        x = self.pos_encoding(x)
        tgt_len = x.size(1)
        tgt_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float('-inf'), device=x.device),
            diagonal=1
        )
        x = self.decoder(x, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(x)
        return logits  # [B, T, vocab]


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

        self.encoder = EncoderCNN(in_channels, self.hidden_dim)
        self.decoder = DecoderTransformer(
            vocab_size=self.vocab_size,
            input_dim=self.hidden_dim,
            num_layers=int(prm.get('layers', 4)),
            num_heads=int(prm.get('heads', 8)),
            dropout=float(prm.get('dropout', 0.1))
        )

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm.get('momentum', 0.9), 0.999)
        )

    def _normalize_captions(self, y: Tensor) -> Tensor:
        if y.ndim == 3:
            if y.size(1) == 1:
                y = y[:, 0, :]
            else:
                y = y[:, :, 0]
        return y.long()

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        steps = 0
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._normalize_captions(captions)

            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.encoder(images)
            logits = self.decoder(inp, memory)

            loss = self.criteria[0](
                logits.reshape(-1, self.vocab_size),
                tgt.reshape(-1)
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
        x: Tensor,
        y: Optional[Tensor] = None,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
        *args,
        **kwargs
    ):
        images = x.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)

        if y is not None:
            captions = y.to(self.device)
            captions = self._normalize_captions(captions)
            inp = captions[:, :-1]
            logits = self.decoder(inp, memory)
            return logits, None

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
            next_tok = logits_step[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
        return generated, None
