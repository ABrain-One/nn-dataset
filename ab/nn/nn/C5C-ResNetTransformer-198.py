import torch
import torch.nn as nn
from typing import Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int = 640, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 640, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(640, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool1(x)
        x = self.relu4(self.conv4(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x.unsqueeze(1)  # [B, 1, H]


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
        self.hidden_dim = int(prm.get("hidden_dim", 640))

        self.encoder = Encoder(hidden_dim=self.hidden_dim, input_channels=in_channels)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
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

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        steps = 0
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            memory = self.encoder(images)  # [B, 1, H]
            context = memory.mean(dim=1, keepdim=True)  # [B, 1, H]
            emb = self.embedding(inp)  # [B, T-1, H]
            ctx = context.expand(-1, emb.size(1), -1)
            x = torch.cat([emb, ctx], dim=-1)

            out, _ = self.gru(x)
            logits = self.fc_out(out)

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

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        context = memory.mean(dim=1, keepdim=True)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inp = captions[:, :-1]
            emb = self.embedding(inp)
            ctx = context.expand(-1, emb.size(1), -1)
            x = torch.cat([emb, ctx], dim=-1)
            out, hidden_state = self.gru(x, hidden_state)
            logits = self.fc_out(out)
            return logits, hidden_state

        batch_size = images.size(0)
        max_len = 20
        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None
        for _ in range(max_len - 1):
            emb = self.embedding(generated[:, -1:])
            ctx = context
            x = torch.cat([emb, ctx], dim=-1)
            out, hidden_state = self.gru(x, hidden_state)
            logits_step = self.fc_out(out[:, -1, :])
            next_tok = logits_step.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
