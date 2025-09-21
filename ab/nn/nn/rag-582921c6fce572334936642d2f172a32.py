# Auto-generated single-file for ClassBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional, Callable
import collections
from itertools import repeat
from functools import partial
from typing import Optional
from typing import Callable
from collections import *

# ---- timm.models.volo.ClassAttention ----
class ClassAttention(nn.Module):
    """Class attention mechanism for class token interaction."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """Initialize ClassAttention.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head. If None, computed as dim // num_heads.
            qkv_bias: Whether to use bias in QKV projection.
            attn_drop: Attention dropout rate.
            proj_drop: Projection dropout rate.
        """
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, self.head_dim * self.num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C) where first token is class token.

        Returns:
            Class token output of shape (B, 1, C).
        """
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim) * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed

# ---- timm.layers.drop.drop_path ----
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

# ---- timm.layers.drop.DropPath ----
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.layers.mlp.Mlp ----
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# ---- ClassBlock (target) ----
class ClassBlock(nn.Module):
    """Class block that combines class attention with MLP."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """Initialize ClassBlock.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head. If None, computed as dim // num_heads.
            mlp_ratio: Ratio for MLP hidden dimension.
            qkv_bias: Whether to use bias in QKV projection.
            drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth drop rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C) where first token is class token.

        Returns:
            Output tensor with updated class token.
        """
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path1(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path2(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)

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
        self.class_block = ClassBlock(
            dim=32,
            num_heads=4,
            head_dim=None,
            mlp_ratio=2.0,
            qkv_bias=False,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
        )
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_token = torch.zeros(B, 1, C, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = self.class_block(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
