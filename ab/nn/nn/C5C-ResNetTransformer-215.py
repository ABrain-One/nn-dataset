import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)                # [B,512]
        x = self.proj(x)                    # [B,H]
        return x.view(x.size(0), 1, -1)     # [B,1,H]


class CustomDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        inputs: torch.Tensor,   # [B,T] indices
        hidden_state: Optional[torch.Tensor],
        features: torch.Tensor,  # [B,S,H]
    ):
        emb = self.embeddings(inputs)       # [B,T,H]
        emb = self.pos_encoder(emb)         # [B,T,H]
        tgt_mask = self._causal_mask(inputs.size(1), inputs.device)  # [T,T]
        out = self.transformer(emb, features, tgt_mask=tgt_mask)     # [B,T,H]
        logits = self.fc_out(out)           # [B,T,V]
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[0])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = 768
        self.num_layers = 6
        self.num_heads = 8

        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=self.hidden_dim)
        self.decoder = CustomDecoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
        )

    def init_zero_hidden(self):
        return None

    def train_setup(self, prm, **kwargs):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm["lr"],
            betas=(prm.get("momentum", 0.9), 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, images, captions, **kwargs):
        # single batch version for compatibility
        images = images.to(self.device, dtype=torch.float32)
        captions = captions.to(self.device)
        captions = self._norm_caps(captions)

        logits, _ = self.forward(images, captions)
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            captions[:, 1:].reshape(-1),
        )
        return loss

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        max_length: int = 20,
        **kwargs,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B,1,H]

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)  # [B,T]
            inp = captions[:, :-1]                # [B,T-1]
            logits, hidden_state = self.decoder(inp, hidden_state, memory)
            return logits, hidden_state

        # Inference: greedy decode
        batch_size = images.size(0)
        sos_idx = 1
        gen = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None
        for _ in range(max_length - 1):
            logits, hidden_state = self.decoder(gen, hidden_state, memory)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, next_tok], dim=1)
        return gen, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
