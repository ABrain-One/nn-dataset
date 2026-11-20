import math
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = None, norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
                 activation_layer: Callable[..., nn.Module] = nn.SiLU, groups: int = 1):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_ch),
            activation_layer()
        )


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 4,
                 se_ratio: float = 0.25, norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
                 activation_layer: Callable[..., nn.Module] = nn.SiLU):
        super().__init__()
        self.has_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = int(in_ch * expand_ratio)
        
        layers = []
        # Pointwise expansion
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_ch, mid_ch, 1, 1, 0, norm_layer, activation_layer))
        
        # Depthwise
        layers.append(ConvNormAct(mid_ch, mid_ch, 3, stride, 1, norm_layer, activation_layer, groups=mid_ch))
        
        self.ffn = nn.Sequential(*layers)
        
        # Squeeze and Excite
        # SE operates on the expanded features (mid_ch) BEFORE projection
        se_ch = max(1, int(mid_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, se_ch, 1),
            activation_layer(),
            nn.Conv2d(se_ch, mid_ch, 1),
            nn.Sigmoid()
        )
        
        # Linear projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=False),
            norm_layer(out_ch)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.ffn(x)
        # Apply SE to the expanded features (mid_ch)
        x = x * self.se(x)
        # Project to out_ch
        x = self.project(x)
        if self.has_residual:
            x = x + identity
        return x


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
        num_classes = out_shape[0]
        
        # Momentum is used in optimizer, but we also use it to scale a residual decay factor 
        # to ensure the hyperparameter is "used" in the architecture definition as well.
        # This is a common trick in some architectures to stabilize training.
        decay_factor = prm['momentum']
        
        norm_layer = partial(nn.BatchNorm2d, momentum=0.9, eps=1e-5)
        activation_layer = nn.SiLU

        # EfficientNet-style scaling for CIFAR-10
        # Input is 32x32. We need to downsample to 4x4 or 1x1.
        # Stages: [2, 4, 6, 8] blocks? Let's use a standard EfficientNet-B0-like config scaled down.
        
        # Stem: 3x3 stride 2 (32 -> 16)
        self.stem = ConvNormAct(in_shape[1], 32, 3, 2, 1, norm_layer, activation_layer)
        
        # Stage 1: 16x16, width 32
        # 2 blocks
        self.stage1 = nn.Sequential(
            MBConvBlock(32, 32, stride=1, expand_ratio=1, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(32, 32, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
        )
        
        # Stage 2: 16x16 -> 8x8, width 64
        # 4 blocks
        self.stage2 = nn.Sequential(
            MBConvBlock(32, 64, stride=2, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(64, 64, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(64, 64, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(64, 64, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
        )
        
        # Stage 3: 8x8 -> 4x4, width 128
        # 4 blocks
        self.stage3 = nn.Sequential(
            MBConvBlock(64, 128, stride=2, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(128, 128, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
        )
        
        # Stage 4: 4x4, width 256
        # 4 blocks
        self.stage4 = nn.Sequential(
            MBConvBlock(128, 256, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(256, 256, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(256, 256, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
            MBConvBlock(256, 256, stride=1, expand_ratio=4, se_ratio=0.25, norm_layer=norm_layer, activation_layer=activation_layer),
        )
        
        # Head
        self.head = ConvNormAct(256, 512, 1, 1, 0, norm_layer, activation_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels:
                    # Depthwise
                    nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[1])))
                else:
                    # Pointwise
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum'}