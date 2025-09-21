# Auto-generated single-file for ChannelBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
import collections
from itertools import repeat
from functools import partial
from collections import *

# ---- timm.models.davit.ChannelAttention ----
class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        k = k * self.scale
        attn = k.transpose(-1, -2) @ v
        attn = attn.softmax(dim=-1)
        x = (attn @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# ---- timm.models.davit.ChannelAttentionV2 ----
class ChannelAttentionV2(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, dynamic_scale=True):
        super().__init__()
        self.groups = num_heads
        self.head_dim = dim // num_heads
        self.dynamic_scale = dynamic_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.dynamic_scale:
            q = q * N ** -0.5
        else:
            q = q * self.head_dim ** -0.5
        attn = q.transpose(-1, -2) @ k
        attn = attn.softmax(dim=-1)
        x = (attn @ v.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# ---- timm.models.davit.ConvPosEnc ----
class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim,
        )
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: Tensor):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x

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

# ---- ChannelBlock (target) ----
class ChannelBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
            v2=False,
    ):
        super().__init__()

        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        attn_layer = ChannelAttentionV2 if v2 else ChannelAttention
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)

        if self.ffn:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path2 = None

    def forward(self, x: Tensor):
        B, C, H, W = x.shape

        x = self.cpe1(x).flatten(2).transpose(1, 2)

        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path1(cur)

        x = self.cpe2(x.transpose(1, 2).view(B, C, H, W))

        if self.mlp is not None:
            x = x.flatten(2).transpose(1, 2)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = x.transpose(1, 2).view(B, C, H, W)

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
        self.channel_block = ChannelBlock(dim=32, num_heads=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = self.channel_block(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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
