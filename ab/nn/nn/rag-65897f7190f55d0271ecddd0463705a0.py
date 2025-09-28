# Auto-generated single-file for ResidualAttentionBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from collections import OrderedDict
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass
from torch.nn import LayerNorm
from typing import Union
from typing import Tuple
from typing import Optional

# ---- mmpretrain.models.utils.clip_generator_helper.QuickGELU ----
class QuickGELU(nn.Module):
    """A faster version of GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return x * torch.sigmoid(1.702 * x)

# ---- ResidualAttentionBlock (target) ----
class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block (RAB).

    This module implements the same function as the MultiheadAttention,
    but with a different interface, which is mainly used
    in CLIP.

    Args:
        d_model (int): The feature dimension.
        n_head (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: Optional[torch.Tensor] = None,
                 return_attention: bool = False) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.return_attention = return_attention

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention function."""
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        if self.return_attention:
            return self.attn(
                x,
                x,
                x,
                need_weights=self.return_attention,
                attn_mask=self.attn_mask)
        else:
            return self.attn(
                x,
                x,
                x,
                need_weights=self.return_attention,
                attn_mask=self.attn_mask)[0]

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward function."""
        if self.return_attention:
            x_, attention = self.attention(self.ln_1(x))
            x = x + x_
            x = x + self.mlp(self.ln_2(x))
            return x, attention
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.residual_attention_block = ResidualAttentionBlock(d_model=64, n_head=8, attn_mask=None, return_attention=False)
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        x = self.residual_attention_block(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
