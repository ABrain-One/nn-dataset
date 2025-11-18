import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)          # [B, H]
        x = x.unsqueeze(1)        # [B, 1, H]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int,
                 num_heads: int = 8, num_layers: int = 2, max_len: int = 52):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = inputs.shape
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.embedding(inputs) + self.pos_embedding(positions)
        tgt_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=inputs.device), 1
        )
        out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict,
                 device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        # out_shape usually like (vocab_size, 1) or (vocab_size,)
        self.vocab_size = out_shape[0][0] if isinstance(out_shape, (tuple, list)) and isinstance(out_shape[0], (tuple, list)) else int(out_shape[0])
        self.hidden_dim = max(int(prm.get('hidden_dim', 640)), 640)

        self.encoder = CNNEncoder(self.in_channels, self.hidden_dim)
        self.decoder = TransformerDecoder(self.vocab_size, self.hidden_dim)

        self.to(self.device)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm.get('momentum', 0.9), 0.999),
        )

    def _normalize_captions(self, captions: torch.Tensor) -> torch.Tensor:
        if captions.dim() == 3:
            captions = captions[:, 0, :]
        return captions.long()

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = self._normalize_captions(captions).to(self.device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)
            logits = self.decoder(inputs, memory)

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size),
                                    targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor,
                captions: Optional[torch.Tensor] = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)

        if captions is not None:
            captions = self._normalize_captions(captions).to(self.device)
            inputs = captions[:, :-1]
            logits = self.decoder(inputs, memory)
            return logits, None

        bsz = images.size(0)
        sos = torch.full((bsz, 1), 1, dtype=torch.long, device=self.device)
        generated = sos
        for _ in range(50):
            logits = self.decoder(generated, memory)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated, None
