import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Iterable, Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Infer dimensions robustly
        self.in_features = self._infer_in_features(in_shape)
        self.num_classes = self._infer_num_classes(out_shape)

        hidden = int(self.prm.get('hidden', 128))
        self.linear1 = nn.Linear(self.in_features, hidden)
        self.linear2 = nn.Linear(hidden, self.num_classes)

        # Created in train_setup()
        self.criterion: Optional[nn.Module] = None
        self.criteria = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # ---------------- training plumbing ----------------
    def train_setup(self, prm: Dict[str, Any]):
        lr = float(prm.get('lr', 1e-3))
        beta1 = float(prm.get('momentum', 0.9))

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]] | Dict[str, torch.Tensor], *_, **__):
        if self.optimizer is None or self.criterion is None:
            self.train_setup(self.prm)

        self.train()

        def _step(data: torch.Tensor, target: torch.Tensor):
            data = data.to(self.device).float()
            target = target.to(self.device).long()
            logits = self.forward(data)
            loss = self.criterion(logits, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        if isinstance(train_data, dict):
            _step(train_data['data'], train_data['target'])
        else:
            for data, target in train_data:
                _step(data, target)

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = x.to(self.device)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # flatten (B, C*H*W)
        assert x.size(1) == self.in_features, f"Expected {self.in_features} features, got {x.size(1)}."

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # ---------------- helpers ----------------
    @staticmethod
    def _infer_in_features(in_shape) -> int:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 4:  # (B, C, H, W)
                return int(in_shape[1]) * int(in_shape[2]) * int(in_shape[3])
            if len(in_shape) == 3:  # (C, H, W)
                return int(in_shape[0]) * int(in_shape[1]) * int(in_shape[2])
            if len(in_shape) == 2:  # (B, D) or (N, D)
                return int(in_shape[1])
            if len(in_shape) == 1:
                return int(in_shape[0])
        return int(in_shape)

    @staticmethod
    def _infer_num_classes(out_shape) -> int:
        if isinstance(out_shape, int):
            return out_shape
        if isinstance(out_shape, (tuple, list)):
            # common patterns: out_shape[0][0] or out_shape[0]
            try:
                return int(out_shape[0][0])
            except Exception:
                try:
                    return int(out_shape[0])
                except Exception:
                    pass
        return int(out_shape)
