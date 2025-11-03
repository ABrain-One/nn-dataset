import torch
import torch.nn as nn
from typing import Optional


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Infer vocab size robustly from out_shape
        self.vocab_size = self._infer_vocab_size(out_shape)

        # Model sizes
        in_channels = int(in_shape[1])
        self.hidden_size = int(prm.get('hidden_size', 512))
        self.max_caption_length = int(prm.get('max_caption_length', 20))
        self.sos_idx = int(prm.get('sos_idx', 1))

        # Simple CNN encoder -> pooled features -> projection
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.hidden_size),
        )

        # Token embedding + LSTM decoder + output head
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=float(prm.get('dropout', 0.0)),
        )
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        # Will be initialized in train_setup
        self.criterion = None
        self.optimizer = None

    @staticmethod
    def _infer_vocab_size(out_shape: tuple) -> int:
        try:
            return int(out_shape[0][0])
        except Exception:
            try:
                return int(out_shape[0])
            except Exception:
                return int(out_shape)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        lr = float(prm.get('lr', 1e-3))
        momentum = float(prm.get('momentum', 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data):
        self.train()
        for i, (images, captions) in enumerate(train_data):
            images = images.to(self.device)           # [B, C, H, W]
            captions = captions.to(self.device)       # [B, L]

            if captions.size(1) < 2:
                continue  # need at least <sos> and one target token

            # Teacher forcing: predict t+1 from tokens up to t
            inputs = captions[:, :-1]                 # [B, L-1]
            targets = captions[:, 1:]                 # [B, L-1]

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.forward(images, inputs)     # [B, L-1, V]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            if i % 300 == 0:
                print(f"Batch {i}: Loss: {float(loss.item()):.4f}")

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images -> [B, H]."""
        return self.encoder(images)

    def decode_sequences(
        self,
        features: torch.Tensor,                     # [B, H]
        captions: Optional[torch.Tensor] = None,    # [B, L] (teacher-forcing inputs) or None for inference
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compatibility wrapper."""
        return self._lstm_decode(features, captions, batch_size)

    def _lstm_decode(
        self,
        features: torch.Tensor,                     # [B, H]
        captions: Optional[torch.Tensor] = None,    # [B, L]
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = features.size(0)

        # Initialize hidden state from features
        h0 = features.unsqueeze(0).contiguous()     # [1, B, H]
        c0 = torch.zeros_like(h0)                   # [1, B, H]
        hidden = (h0, c0)

        if captions is not None:
            # Teacher forcing path
            emb = self.embedding(captions)          # [B, L, H]
            out, _ = self.rnn(emb, hidden)          # [B, L, H]
            logits = self.fc_out(out)               # [B, L, V]
            return logits

        # Inference (greedy)
        inputs = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=features.device)
        logits_steps = []
        for _ in range(self.max_caption_length):
            emb = self.embedding(inputs[:, -1:])    # [B, 1, H]
            out, hidden = self.rnn(emb, hidden)     # [B, 1, H]
            step_logits = self.fc_out(out)          # [B, 1, V]
            logits_steps.append(step_logits)
            next_token = step_logits.argmax(dim=-1) # [B, 1]
            inputs = torch.cat([inputs, next_token], dim=1)
        return torch.cat(logits_steps, dim=1)       # [B, T, V]

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.encode_features(images)     # [B, H]
        return self.decode_sequences(features, captions)
