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


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor,
                init_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(inputs)
        out, state = self.lstm(x, init_state)
        logits = self.fc_out(out)
        return logits, state


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
        self.decoder = LSTMDecoder(self.vocab_size, self.hidden_dim, num_layers=1)

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

            enc = self.encoder(images)      # [B, H]
            h0 = enc.unsqueeze(0)           # [1, B, H]
            c0 = torch.zeros_like(h0)       # [1, B, H]

            logits, _ = self.decoder(inputs, (h0, c0))

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
        enc = self.encoder(images)
        bsz = images.size(0)

        if hidden_state is None:
            h0 = enc.unsqueeze(0)
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = hidden_state

        if captions is not None:
            captions = self._normalize_captions(captions).to(self.device)
            inputs = captions[:, :-1]
            logits, state = self.decoder(inputs, (h0, c0))
            return logits, state

        sos = torch.full((bsz, 1), 1, dtype=torch.long, device=self.device)
        generated = sos
        state = (h0, c0)
        for _ in range(50):
            logits, state = self.decoder(generated[:, -1:], state)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated, state
