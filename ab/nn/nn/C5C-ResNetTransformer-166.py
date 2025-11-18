import torch
import torch.nn as nn
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
        return x


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, init_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(inputs)
        h0 = init_hidden.unsqueeze(0)  # [1, B, H]
        out, hn = self.gru(x, h0)
        logits = self.fc_out(out)
        return logits, hn.squeeze(0)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict,
                 device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = out_shape[0][0] if isinstance(out_shape, (tuple, list)) and isinstance(out_shape[0], (tuple, list)) else int(out_shape[0])
        self.hidden_dim = max(int(prm.get('hidden_dim', 640)), 640)

        self.encoder = CNNEncoder(self.in_channels, self.hidden_dim)
        self.decoder = GRUDecoder(self.vocab_size, self.hidden_dim, num_layers=1)

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

            enc = self.encoder(images)              # [B, H]
            logits, _ = self.decoder(inputs, enc)   # [B, T-1, V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size),
                                    targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor,
                captions: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None):
        images = images.to(self.device, dtype=torch.float32)
        enc = self.encoder(images)

        if captions is not None:
            captions = self._normalize_captions(captions).to(self.device)
            inputs = captions[:, :-1]
            logits, h = self.decoder(inputs, enc)
            return logits, h

        bsz = images.size(0)
        sos = torch.full((bsz, 1), 1, dtype=torch.long, device=self.device)
        generated = sos
        state = enc
        for _ in range(50):
            logits, state = self.decoder(generated[:, -1:], state)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated, state
