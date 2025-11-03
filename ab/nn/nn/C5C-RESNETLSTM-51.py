import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable, Union


def supported_hyperparameters():
    return {"lr", "momentum"}


# --------- small helpers ---------
class FlattenLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class BBBConv2d(nn.Conv2d):
    """Placeholder 'Bayesian' conv that just calls the parent conv.
    Keeps a `priors` field for API compatibility.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        priors=None,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.priors = priors

    def forward(self, x: Tensor) -> Tensor:
        # If priors were used, you'd modify weights/activations here.
        return super().forward(x)


class BBBLinear(nn.Linear):
    """Placeholder 'Bayesian' linear that just calls the parent linear."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, priors=None):
        super().__init__(in_features, out_features, bias=bias)
        self.priors = priors

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


# --------- main model ---------
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # Resolve channels / vocab size robustly
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1 else 3

        def _first_int(x):
            return _first_int(x[0]) if isinstance(x, (tuple, list)) else int(x)

        self.vocab_size = _first_int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Optional priors for BBB layers
        self.priors = (prm or {}).get("priors", None)

        # --- Encoder (AlexNet-ish, ends with global pool -> 128 -> 768) ---
        self.conv1 = BBBConv2d(self.in_channels, 64, kernel_size=11, stride=4, padding=5, bias=True, priors=self.priors)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(64, 192, kernel_size=5, padding=2, bias=True, priors=self.priors)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(192, 384, kernel_size=3, padding=1, bias=True, priors=self.priors)
        self.conv4 = BBBConv2d(384, 256, kernel_size=3, padding=1, bias=True, priors=self.priors)
        self.conv5 = BBBConv2d(256, 128, kernel_size=3, padding=1, bias=True, priors=self.priors)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_lin = BBBLinear(128, 768, bias=True, priors=self.priors)

        # --- Simple decoder head ---
        self.classifier = nn.Linear(768, self.vocab_size)

        # Generation settings
        self.max_len = int((prm or {}).get("max_len", 20))
        self.sos_idx = int((prm or {}).get("sos_idx", 1))
        self.eos_idx = int((prm or {}).get("eos_idx", self.vocab_size - 1))

        # Training utils (initialized in train_setup but provide defaults)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float((prm or {}).get("lr", 1e-4)),
            betas=(float((prm or {}).get("momentum", 0.9)), 0.999),
        )

        self.to(self.device)

    # ---- API: training setup ----
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm["lr"]), betas=(float(prm.get("momentum", 0.9)), 0.999)
        )

    # ---- API: single epoch style learner ----
    def learn(self, train_data: Iterable[Tuple[Tensor, Tensor]]):
        self.train()
        for images, captions in train_data:
            if isinstance(captions, tuple):
                captions = captions[0]

            images = images.to(self.device)
            captions = captions.to(self.device)

            # Normalize caption shape to (B, T)
            if captions.dim() == 3:
                # If one-hot or extra dim -> pick indices on last dim or squeeze last axis if singleton
                if captions.size(-1) > 1:
                    captions = captions.argmax(dim=-1)
                else:
                    captions = captions.squeeze(-1)

            logits, _ = self.forward(images, captions)

            # Next-token prediction
            targets = captions[:, 1:] if captions.size(1) > 1 else captions[:, :0]  # handle T=1 safely
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            # Yield scalar for logging if caller consumes it
            yield float(loss.detach().cpu().item())

    # ---- forward ----
    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        images: (B, C, H, W)
        captions: (B, T) token ids; if provided, returns teacher-forced logits (B, T-1, V)
        returns: (logits, None)
        """
        # Encode images into a single vector per image
        x = F.relu(self.conv1(images))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.global_pool(x)          # (B, 128, 1, 1)
        x = x.view(x.size(0), 128)       # (B, 128)
        feats = F.relu(self.final_lin(x))  # (B, 768)

        if captions is not None:
            # Teacher forcing: predict next tokens for all positions
            if captions.size(1) <= 1:
                # No next-token positions: return empty logits with correct dims
                empty = feats.new_zeros((feats.size(0), 0, self.vocab_size))
                return empty, None

            seq_len = captions.size(1) - 1
            ctx = feats.unsqueeze(1).expand(-1, seq_len, -1)   # (B, T-1, 768)
            logits = self.classifier(ctx)                      # (B, T-1, V)
            return logits, None

        # Inference: greedy decode using only image context
        B = images.size(0)
        ys = torch.full((B, 1), self.sos_idx, device=images.device, dtype=torch.long)
        logits_steps = []

        for _ in range(self.max_len):
            ctx = feats.unsqueeze(1)                # (B, 1, 768)
            step_logits = self.classifier(ctx)      # (B, 1, V)
            logits_steps.append(step_logits)
            next_ids = step_logits.argmax(dim=-1)   # (B, 1)
            ys = torch.cat([ys, next_ids], dim=1)
            if (next_ids == self.eos_idx).all():
                break

        logits = torch.cat(logits_steps, dim=1) if logits_steps else feats.new_zeros((B, 0, self.vocab_size))
        return logits, None
