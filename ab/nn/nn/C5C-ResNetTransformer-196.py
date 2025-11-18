import torch
import torch.nn as nn
from typing import Optional, Tuple


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(512, hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = x.flatten(1)
        x = self.proj(x)
        x = x.unsqueeze(1)
        return x  # [B, 1, H]


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
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
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
        self.hidden_size = 768
        self.num_layers = 2
        self.num_heads = 8

        self.encoder = CNNEncoder(in_channels, hidden_size=self.hidden_size)

        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_enc = PositionalEncoding(self.hidden_size, max_len=int(prm.get('max_len', 64)))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers,
        )
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm.get('momentum', 0.9), 0.999),
            weight_decay=prm.get('weight_decay', 1e-5),
        )

    def _normalize_captions(self, captions: torch.Tensor) -> torch.Tensor:
        if captions.ndim == 3:
            if captions.size(1) == 1:
                captions = captions[:, 0, :]
            else:
                captions = captions[:, :, 0]
        return captions.long()

    def _make_tgt_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device),
            diagonal=1,
        )
        return mask

    def learn(self, images, captions):
        self.train()
        images = images.to(self.device)
        captions = captions.to(self.device)
        captions = self._normalize_captions(captions)

        inp = captions[:, :-1]
        tgt = captions[:, 1:]

        memory = self.encoder(images)
        emb = self.embed(inp)
        emb = self.pos_enc(emb)

        tgt_mask = self._make_tgt_mask(emb.size(1), emb.device)
        output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(output)

        loss = self.criteria[0](
            logits.reshape(-1, self.vocab_size),
            tgt.reshape(-1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 3.0)
        self.optimizer.step()
        return loss

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)

        memory = self.encoder(images)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._normalize_captions(captions)

            inp = captions[:, :-1]
            emb = self.embed(inp)
            emb = self.pos_enc(emb)

            tgt_mask = self._make_tgt_mask(emb.size(1), emb.device)
            output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output)
            hidden_state = output[:, -1, :] if hidden_state is None else hidden_state
            return logits, hidden_state

        batch_size = images.size(0)
        max_len = 20
        sos = torch.full((batch_size, 1), 1, dtype=torch.long, device=self.device)
        generated = sos
        hidden_state = None
        for _ in range(max_len - 1):
            emb = self.embed(generated)
            emb = self.pos_enc(emb)
            tgt_mask = self._make_tgt_mask(emb.size(1), emb.device)
            output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
            logits_step = self.fc_out(output[:, -1, :])
            next_tok = logits_step.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            hidden_state = output[:, -1, :]
        return generated, hidden_state
