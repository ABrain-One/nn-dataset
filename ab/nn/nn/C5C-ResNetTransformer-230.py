import torch
import torch.nn as nn
from typing import Any, Iterable, Optional


class CNN_Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, final_channel=768):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.final_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 8, final_channel),
        )
        self.out_channels = final_channel

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final_proj(x)
        return x.unsqueeze(1)


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Any,
        out_shape: Any,
        prm: dict,
        device: torch.device,
        *_,
        **__,
    ):
        super().__init__()
        self.device = device

        if isinstance(in_shape, (tuple, list)):
            in_channels = int(in_shape[1] if len(in_shape) > 1 else in_shape[0])
        else:
            in_channels = int(in_shape)

        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        self.hidden_dim = 768
        self.encoder = CNN_Encoder(in_channels=in_channels, final_channel=self.hidden_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data: Iterable):
        assert self.optimizer is not None
        self.train()
        total_loss = 0.0
        steps = 0

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)
            ctx = memory.mean(dim=1, keepdim=True)

            emb = self.embedding(inputs)
            ctx_exp = ctx.expand(-1, emb.size(1), -1)
            dec_in = torch.cat([emb, ctx_exp], dim=-1)

            outputs, _ = self.gru(dec_in)
            logits = self.fc_out(outputs)

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
        memory = self.encoder(images)
        ctx = memory.mean(dim=1, keepdim=True)
        batch_size = images.size(0)

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]

            emb = self.embedding(inputs)
            ctx_exp = ctx.expand(-1, emb.size(1), -1)
            dec_in = torch.cat([emb, ctx_exp], dim=-1)

            outputs, hidden_state = self.gru(dec_in, hidden_state)
            logits = self.fc_out(outputs)
            return logits, hidden_state

        sos_idx = 1
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        hidden_state = None

        for _ in range(max_length - 1):
            emb = self.embedding(generated[:, -1:])
            ctx_exp = ctx
            dec_in = torch.cat([emb, ctx_exp], dim=-1)
            outputs, hidden_state = self.gru(dec_in, hidden_state)
            step_logits = self.fc_out(outputs[:, -1, :])
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated, hidden_state


def supported_hyperparameters():
    return {"lr", "momentum"}
