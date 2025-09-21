# Auto-generated single-file for Attention2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Dict
import os
import math

# ---- timm.layers.config._EXPORTABLE ----
_EXPORTABLE = False

# ---- timm.layers.config._HAS_FUSED_ATTN ----
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

# ---- timm.layers.config._USE_FUSED_ATTN ----
_USE_FUSED_ATTN = int(os.environ.get('TIMM_FUSED_ATTN', '0'))

# ---- timm.layers.config.use_fused_attn ----
def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0

# ---- Missing dependencies ----
def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding=kernel_size//2, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.norm(self.conv(x))

def ndgrid(*tensors):
    return torch.meshgrid(*tensors, indexing='ij')

# ---- Attention2d (target) ----
class Attention2d(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            dim=384,
            key_dim=32,
            num_heads=8,
            attn_ratio=4,
            resolution=7,
            act_layer=nn.GELU,
            stride=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim

        resolution = to_2tuple(resolution)
        if stride is not None:
            resolution = tuple([math.ceil(r / stride) for r in resolution])
            self.stride_conv = ConvNorm(dim, dim, kernel_size=3, stride=stride, groups=dim)
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.stride_conv = None
            self.upsample = None

        self.resolution = resolution
        self.N = self.resolution[0] * self.resolution[1]
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        kh = self.key_dim * self.num_heads

        self.q = ConvNorm(dim, kh)
        self.k = ConvNorm(dim, kh)
        self.v = ConvNorm(dim, self.dh)
        self.v_local = ConvNorm(self.dh, self.dh, kernel_size=3, groups=self.dh)
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        self.act = act_layer()
        self.proj = ConvNorm(self.dh, dim, 1)

        pos = torch.stack(ndgrid(torch.arange(self.resolution[0]), torch.arange(self.resolution[1]))).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, self.N))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(rel_pos), persistent=False)
        self.attention_bias_cache = {}  # per-device attention_biases cache (data-parallel compat)

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
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn + self.get_attention_biases(x.device)
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v).transpose(2, 3)
        x = x.reshape(B, self.dh, self.resolution[0], self.resolution[1]) + v_local
        if self.upsample is not None:
            x = self.upsample(x)

        x = self.act(x)
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

        self.attention2d = Attention2d(dim=32, key_dim=8, num_heads=4, resolution=7)
        
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
