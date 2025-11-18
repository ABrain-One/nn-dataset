import torch
import torch.nn as nn
from typing import Optional, Tuple


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNEncoder(nn.Module):
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

            nn.Conv2d(256, hidden_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(2).transpose(1, 2)
        return x  # [B, S, H]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
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
        return x + self.pe[:, :seq_len]


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
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

        self.encoder = CNNEncoder(in_channels, hidden_dim=self.hidden_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim, max_len=int(prm.get('max_len', 64)))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(prm.get('heads', 8)),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(prm.get('dropout', 0.1)),
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(prm.get('layers', 4)),
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, hyperparameters):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hyperparameters['lr'],
            betas=(hyperparameters.get('momentum', 0.9), 0.999),
        )

    def _normalize_captions(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 3:
            if y.size(1) == 1:
                y = y[:, 0, :]
            else:
                y = y[:, :, 0]
        return y.long()

    def _make_tgt_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device),
            diagonal=1,
        )
        return mask

    def learn(self, train_data, hyperparameters=None):
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

            emb = self.embedding(inp)
            emb = self.pos_encoding(emb)
            tgt_mask = self._make_tgt_mask(emb.size(1), emb.device)
            dec_out = self.decoder(emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(dec_out)

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

    def forward(self, x, y=None, **kwargs):
        images = x.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)

        if y is not None:
            captions = y.to(self.device)
            captions = self._normalize_captions(captions)
            inp = captions[:, :-1]

            emb = self.embedding(inp)
            emb = self.pos_encoding(emb)
            tgt_mask = self._make_tgt_mask(emb.size(1), emb.device)
            dec_out = self.decoder(emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(dec_out)
            return logits, None

        batch_size = images.size(0)
        max_len = 20
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1),
            sos_idx,
            dtype=torch.long,
            device=self.device,
        )
        for _ in range(max_len - 1):
            emb = self.embedding(generated)
            emb = self.pos_encoding(emb)
            tgt_mask = self._make_tgt_mask(emb.size(1), emb.device)
            dec_out = self.decoder(emb, memory, tgt_mask=tgt_mask)
            next_tok = self.fc_out(dec_out[:, -1, :]).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, None
