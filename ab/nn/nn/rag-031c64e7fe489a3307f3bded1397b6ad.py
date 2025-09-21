# Auto-generated single-file for AttentionDownsample
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Dict, Tuple
import collections
from collections import OrderedDict
from itertools import repeat
from functools import partial
from typing import Dict
from collections import *
from typing import Tuple

# ---- timm.models.levit.ConvNorm ----
class ConvNorm(nn.Module):
    def __init__(
            self, in_chs, out_chs, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)

        nn.init.constant_(self.bn.weight, bn_weight_init)

    def fuse(self):
        c, bn = self.linear, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.linear.stride,
            padding=self.linear.padding, dilation=self.linear.dilation, groups=self.linear.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        return self.bn(self.linear(x))

# ---- timm.models.levit.LinearNorm ----
class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

        nn.init.constant_(self.bn.weight, bn_weight_init)

    def fuse(self):
        l, bn = self.linear, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        x = self.linear(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)

# ---- timm.layers.grid.ndgrid ----
def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.models.levit.Downsample ----
class Downsample(nn.Module):
    def __init__(self, stride, resolution, use_pool=False):
        super().__init__()
        self.stride = stride
        self.resolution = to_2tuple(resolution)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False) if use_pool else None

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution[0], self.resolution[1], C)
        if self.pool is not None:
            x = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            x = x[:, ::self.stride, ::self.stride]
        return x.reshape(B, -1, C)

# ---- AttentionDownsample (target) ----
class AttentionDownsample(nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            in_dim,
            out_dim,
            key_dim,
            num_heads=8,
            attn_ratio=2.0,
            stride=2,
            resolution=14,
            use_conv=False,
            use_pool=False,
            act_layer=nn.SiLU,
    ):
        super().__init__()
        resolution = to_2tuple(resolution)

        self.stride = stride
        self.resolution = resolution
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * self.num_heads
        self.scale = key_dim ** -0.5
        self.use_conv = use_conv

        if self.use_conv:
            ln_layer = ConvNorm
            sub_layer = partial(
                nn.AvgPool2d,
                kernel_size=3 if use_pool else 1, padding=1 if use_pool else 0, count_include_pad=False)
        else:
            ln_layer = LinearNorm
            sub_layer = partial(Downsample, resolution=resolution, use_pool=use_pool)

        self.kv = ln_layer(in_dim, self.val_attn_dim + self.key_attn_dim)
        self.q = nn.Sequential(OrderedDict([
            ('down', sub_layer(stride=stride)),
            ('ln', ln_layer(in_dim, self.key_attn_dim))
        ]))
        self.proj = nn.Sequential(OrderedDict([
            ('act', act_layer()),
            ('ln', ln_layer(self.val_attn_dim, out_dim))
        ]))

        self.attention_biases = nn.Parameter(torch.zeros(num_heads, resolution[0] * resolution[1]))
        k_pos = torch.stack(ndgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))).flatten(1)
        q_pos = torch.stack(ndgrid(
            torch.arange(0, resolution[0], step=stride),
            torch.arange(0, resolution[1], step=stride)
        )).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.register_buffer('attention_bias_idxs', rel_pos, persistent=False)

        self.attention_bias_cache = {}  # per-device attention_biases cache

    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            HH, WW = (H - 1) // self.stride + 1, (W - 1) // self.stride + 1
            k, v = self.kv(x).view(B, self.num_heads, -1, H * W).split([self.key_dim, self.val_dim], dim=2)
            q = self.q(x).view(B, self.num_heads, self.key_dim, -1)

            attn = (q.transpose(-2, -1) @ k) * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (v @ attn.transpose(-2, -1)).reshape(B, self.val_attn_dim, HH, WW)
        else:
            B, N, C = x.shape
            k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
            k = k.permute(0, 2, 3, 1)  # BHCN
            v = v.permute(0, 2, 1, 3)  # BHNC
            q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)

            attn = q @ k * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
        x = self.proj(x)
        return x


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self.attention_downsample = AttentionDownsample(
            in_dim=32, out_dim=32, key_dim=8, num_heads=4, 
            stride=1, resolution=7, use_conv=True
        )
        
        layers += [
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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
