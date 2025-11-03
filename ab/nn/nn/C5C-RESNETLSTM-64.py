import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Iterable


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Infer channels and output dimension robustly
        self.in_channels = int(in_shape[1]) if len(in_shape) >= 2 else int(in_shape[0])

        def _first_int(x: Any) -> int:
            while isinstance(x, (tuple, list)):
                x = x[0]
            return int(x)

        self.out_dim = _first_int(out_shape)

        # Encoder configuration (simple pyramid of conv blocks)
        encoder_depth = 4
        growth_rates = [64, 128, 256, 512]
        block_sizes = [encoder_depth] * len(growth_rates)

        self.cnn = nn.ModuleList()
        prev_in = self.in_channels
        for block_size, gr in zip(block_sizes, growth_rates):
            for _ in range(block_size):
                self.cnn.append(
                    nn.Sequential(
                        nn.Conv2d(prev_in, gr, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(gr),
                        nn.ReLU(inplace=True),
                    )
                )
                prev_in = gr

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head: nn.Linear | None = None  # created lazily when input size known

        self.to(self.device)

    # -------- training helpers --------
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-4)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        """Simple supervised loop over (images, targets)."""
        self.train()
        for images, targets in train_data:
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self(images)
            loss = self.criteria[0](logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # -------- forward --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Light sanity check on channels
        assert x.dim() == 4 and x.size(1) == self.in_channels, (
            f"Expected input with {self.in_channels} channels, got {tuple(x.shape)}"
        )

        for block in self.cnn:
            x = block(x)

        x = self.pool(x).flatten(1)  # [B, C]

        # Lazy head creation (in_features depends on encoder output channels)
        if self.head is None:
            self.head = nn.Linear(x.size(1), self.out_dim).to(x.device)

        logits = self.head(x)  # [B, out_dim]
        return logits
