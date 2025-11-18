import torch
import torch.nn as nn
from typing import Optional


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.embed_dim = 768

        # Encoder (keep their idea, just safe)
        in_channels = 3 if not isinstance(in_shape, (tuple, list)) else int(in_shape[0])
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.embed_dim),
        )

        # Decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # input to GRU: token embedding + image embedding
        self.gru = nn.GRU(
            input_size=self.embed_dim * 2,
            hidden_size=self.embed_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(self.embed_dim, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm["lr"],
            momentum=prm.get("momentum", 0.9),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)  # [B,T]

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)  # [B,H]
            emb = self.embedding(inputs)  # [B,T-1,H]
            mem_expanded = memory.unsqueeze(1).expand(-1, emb.size(1), -1)
            dec_inp = torch.cat([emb, mem_expanded], dim=-1)  # [B,T-1,2H]

            output, _ = self.gru(dec_inp)  # [B,T-1,H]
            logits = self.fc(output)       # [B,T-1,V]

            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
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
        hidden_state: Optional[torch.Tensor] = None,
        max_length: int = 20,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # [B,H]

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]  # [B,T-1]

            emb = self.embedding(inputs)
            mem_expanded = memory.unsqueeze(1).expand(-1, emb.size(1), -1)
            dec_inp = torch.cat([emb, mem_expanded], dim=-1)

            output, hidden_state = self.gru(dec_inp, hidden_state)
            logits = self.fc(output)
            return logits, hidden_state

        # Inference
        batch_size = images.size(0)
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            emb = self.embedding(generated[:, -1:])
            mem_expanded = memory.unsqueeze(1)
            dec_inp = torch.cat([emb, mem_expanded], dim=-1)
            output, hidden_state = self.gru(dec_inp, hidden_state)
            logits = self.fc(output[:, -1, :])  # [B,V]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
