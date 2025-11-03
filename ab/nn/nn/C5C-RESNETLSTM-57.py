import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


class Net(nn.Module):
    """
    Minimal, compilable model that matches the expected API:
      - __init__(in_shape, out_shape, prm, device, *_, **__)
      - train_setup(prm)
      - learn(train_data)
      - forward(x, y=None)
    """

    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm or {}

        # Infer input feature size from in_shape (supports (B,C,H,W) or (C,H,W) or (N,))
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 3:
                # Use last 3 dims as (C, H, W)
                c, h, w = int(in_shape[-3]), int(in_shape[-2]), int(in_shape[-1])
                in_features = c * h * w
            elif len(in_shape) == 1:
                in_features = int(in_shape[0])
            else:
                raise ValueError(f"Unsupported in_shape: {in_shape}")
        else:
            in_features = int(in_shape)

        # Infer output dimension from out_shape (first element)
        if isinstance(out_shape, (tuple, list)):
            if len(out_shape) == 0:
                raise ValueError("out_shape is empty; cannot infer output size.")
            out_dim = int(out_shape[0])
        else:
            out_dim = int(out_shape)

        self.in_features = in_features
        self.out_dim = out_dim

        # A simple MLP head
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.out_dim),
        )

        self.to(self.device)

    def train_setup(self, prm):
        """Setup optimizer and loss (keeps the expected tuple form for criteria)."""
        self.to(self.device)
        lr = float(prm.get("lr", 1e-3))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(momentum, 0.999))

    def learn(self, train_data):
        """
        Minimal training step.
        Expects either:
          - a tuple (inputs, targets), or
          - an iterable of such tuples.
        Uses CE if targets are integer class labels; otherwise falls back to MSE.
        """
        self.train()

        def _one_step(inputs, targets):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.forward(inputs)

            # Choose a reasonable loss automatically
            if targets.dtype in (torch.int64, torch.long) and outputs.dim() == 2 and outputs.size(1) > 1:
                loss = self.criteria[0](outputs, targets)
            else:
                loss = F.mse_loss(outputs, targets.float())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            return float(loss.detach().cpu())

        # If a single (inputs, targets) pair is provided
        if isinstance(train_data, tuple) and len(train_data) == 2:
            return _one_step(*train_data)

        # Otherwise, iterate
        last_loss = None
        for inputs, targets in train_data:
            last_loss = _one_step(inputs, targets)
        return last_loss if last_loss is not None else 0.0

    def forward(self, x, y=None):
        """
        Forward pass. 'y' is unused here but kept for API compatibility with
        teacher-forcing style signatures in other models.
        """
        return self.model(x)
