import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation, MLP


class MLPBlock(MLP):
    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i + 1}.{type}"
                    new_key = f"{prefix}{3 * i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'patch_size'}


def get_divisors(n, res=None):
    res = res or []
    i = 1
    while i <= n:
        if n % i == 0:
            res.append(i)
        i = i + 1
    return res


def get_closest_split(n, close_to):
    all_divisors = get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if close_to < val:
            if ix == 0: return val
            if (val - close_to) > (close_to - all_divisors[ix - 1]):
                return all_divisors[ix - 1]
            return val


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, use_se=False, drop_path_rate=0.0):
        super().__init__()
        self.use_se = use_se
        self.drop_path_rate = drop_path_rate
        
        # 3x3 Conv -> Norm -> Activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        # Optional Dropout2d
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Optional SE
        if use_se:
            self.se = SE(out_channels)
        else:
            self.se = nn.Identity()

        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        if self.use_se:
            out = self.se(out)
            
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        identity = self.shortcut(identity)
        
        out = out + identity
        
        if self.drop_path_rate > 0:
            out = drop_path(out, self.drop_path_rate)
            
        return out

class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        image_size: int = in_shape[2]
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        
        # Use patch_size and attention_dropout to determine configuration choices
        # patch_size > 0.5 implies larger receptive field/complexity -> use SE
        use_se = prm['patch_size'] > 0.5
        # attention_dropout > 0.05 implies need for regularization -> use stochastic depth
        use_drop_path = prm['attention_dropout'] > 0.05
        drop_path_rate = 0.1 if use_drop_path else 0.0

        # Standard CIFAR ResNet configuration
        # Stem: 3x3 Conv, Norm, ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32, 64 channels
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, dropout=dropout, use_se=use_se, drop_path_rate=drop_path_rate),
            ResidualBlock(64, 64, stride=1, dropout=dropout, use_se=use_se, drop_path_rate=drop_path_rate),
        )

        # Stage 2: 16x16, 128 channels
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=dropout, use_se=use_se, drop_path_rate=drop_path_rate),
            ResidualBlock(128, 128, stride=1, dropout=dropout, use_se=use_se, drop_path_rate=drop_path_rate),
        )

        # Stage 3: 8x8, 256 channels
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout=dropout, use_se=use_se, drop_path_rate=drop_path_rate),
            ResidualBlock(256, 256, stride=1, dropout=dropout, use_se=use_se, drop_path_rate=drop_path_rate),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x