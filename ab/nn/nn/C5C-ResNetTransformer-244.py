import torch
import torch.nn as nn
from typing import Optional, Tuple, Iterable, Any


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Tuple,
        out_shape: Tuple,
        prm: dict,
        device: torch.device,
        *_,
        **__,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        vocab_size = int(out_shape[0])

        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 768),
        )

        self.embedding = nn.Embedding(vocab_size, 768)
        self.gru = nn.GRU(768, 768, batch_first=True)
        self.fc_out = nn.Linear(768, vocab_size)

        self.criterion: nn.Module = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=0.1,
        )
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )
        self.criterion = self.criterion.to(self.device)

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
        losses = []

        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)

            memory = self.encoder(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            embedded = self.embedding(inputs)
            output, _ = self.gru(embedded)
            logits = self.fc_out(output)

            vocab_size = int(self.out_shape[0])
            loss = self.criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        max_length: int = 20,
    ):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        batch_size = images.size(0)
        vocab_size = int(self.out_shape[0])

        if captions is not None:
            captions = captions.to(self.device)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]

            embedded = self.embedding(inputs)
            if hidden_state is None:
                hidden_state = torch.zeros(
                    1,
                    batch_size,
                    768,
                    device=self.device,
                )
            output, hidden_state = self.gru(embedded, hidden_state)
            logits = self.fc_out(output)
            return logits, hidden_state

        sos_idx = 1
        generated = torch.full(
            (batch_size, 1),
            sos_idx,
            dtype=torch.long,
            device=self.device,
        )
        hidden_state = torch.zeros(1, batch_size, 768, device=self.device)

        for _ in range(max_length - 1):
            embedded = self.embedding(generated[:, -1:])
            output, hidden_state = self.gru(embedded, hidden_state)
            step_logits = self.fc_out(output[:, -1, :])
            next_token = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        assert generated.dim() == 2
        assert generated.size(1) == max_length
        assert generated.max().item() < vocab_size

        return generated, hidden_state
