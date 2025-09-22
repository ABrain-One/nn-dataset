# Auto-generated single-file for CascadedGroupAttention
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# dws, qkv
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from typing import Dict
import itertools

# ---- timm.models.efficientvit_msra.ConvNorm ----
class ConvNorm(torch.nn.Sequential):
    def __init__(self, in_chs, out_chs, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.conv.groups, w.size(0), w.shape[2:],
            stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# ---- CascadedGroupAttention (target) ----
class CascadedGroupAttention(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
            kernels=(5, 5, 5, 5),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(ConvNorm(dim // (num_heads), self.key_dim * 2 + self.val_dim))
            dws.append(ConvNorm(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            ConvNorm(self.val_dim * num_heads, dim, bn_weight_init=0)
        )

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N), persistent=False)
        self.attention_bias_cache = {}

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
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        attn_bias = self.get_attention_biases(x.device)
        for head_idx, (qkv, dws) in enumerate(zip(self.qkvs, self.dws)):
            if head_idx > 0:
                feat = feat + feats_in[head_idx]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.val_dim], dim=1)
            q = dws(q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            q = q * self.scale
            attn = q.transpose(-2, -1) @ k
            attn = attn + attn_bias[head_idx]
            attn = attn.softmax(dim=-1)
            feat = v @ attn.transpose(-2, -1)
            feat = feat.view(B, self.val_dim, H, W)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self.attention = CascadedGroupAttention(dim=32, key_dim=8, num_heads=2, attn_ratio=2, resolution=16)

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        x = self.attention(x)
        x = torch.nn.functional.interpolate(x, size=(1, 1), mode='bilinear', align_corners=False)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
