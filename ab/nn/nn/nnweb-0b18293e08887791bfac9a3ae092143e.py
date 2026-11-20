import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BasicBlock(nn.Module):
    """
    Basic Residual Block:
    3x3 Conv -> Norm -> Act -> [Dropout] -> [SE]
    3x3 Conv -> Norm -> Act -> [Dropout] -> [SE]
    Shortcut: Identity or 1x1 Conv + Norm
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        use_se: bool = True,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = activation()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
        # Squeeze and Excitation
        if use_se:
            se_reduction = max(out_channels // 16, 1)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, se_reduction),
                activation(),
                nn.Linear(se_reduction, out_channels),
                nn.Sigmoid(),
            )
        else:
            self.se = nn.Identity()
            
        # Stochastic Depth probability
        self.stochastic_depth_prob = stochastic_depth_prob

        # Shortcut
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Apply SE
        if not isinstance(self.se, nn.Identity):
            se_out = self.se(out)
            # Reshape to [B, C, 1, 1] to match spatial dimensions of out
            se_out = se_out.view(out.size(0), out.size(1), 1, 1)
            out = out * se_out
            
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        # Apply Stochastic Depth
        if self.training and self.stochastic_depth_prob > 0:
            # Sample a binary mask for each sample in the batch
            keep_prob = 1.0 - self.stochastic_depth_prob
            # Shape: [B, 1, 1, 1]
            mask = torch.bernoulli(torch.full((out.size(0), 1, 1, 1), keep_prob, device=out.device))
            out = out * mask
            
        return out


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
        input_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Hyperparameters
        dropout = prm['dropout']
        attention_dropout = prm['attention_dropout']
        stochastic_depth_prob = prm['stochastic_depth_prob']
        momentum = prm['momentum'] # Used in optimizer and norm layers
        
        # Architecture details
        # Stem: 3x3 Conv, Norm, Act
        # Stage 1: Res blocks at 32x32
        # Stage 2: Res blocks at 16x16
        # Stage 3: Res blocks at 8x8
        
        # We need to derive spatial sizes from in_shape.
        # CIFAR-10 is 32x32.
        # Stem: 3x3 Conv, stride 1? Or stride 2?
        # "Stem: 3x3 Conv2d, norm, activation."
        # Usually CIFAR ResNets use stride 1 in stem to keep 32x32.
        # If I use stride 1, the output is 32x32.
        # Stage 1: 32x32.
        # Stage 2: 16x16 (requires downsampling).
        # Stage 3: 8x8 (requires downsampling).
        
        # Let's assume standard CIFAR ResNet configuration:
        # Stem: 3x3, stride 1, padding 1. Output 32x32.
        # Stage 1: N blocks, stride 1. Output 32x32.
        # Stage 2: N blocks, stride 2 (first block). Output 16x16.
        # Stage 3: N blocks, stride 2 (first block). Output 8x8.
        
        # Widths:
        # The description doesn't specify widths. "Custom small residual network".
        # Standard CIFAR ResNet-18 uses [64, 128, 256].
        # I will use [64, 128, 256] as the standard choice for CIFAR ResNets.
        
        # Depth:
        # "Custom small". I will use 2 blocks per stage (ResNet-18 style depth for CIFAR).
        
        norm_layer = partial(nn.BatchNorm2d, momentum=prm['momentum'], eps=1e-5)
        activation = nn.ReLU
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            activation()
        )
        
        # Stages
        # Stage 1: 32x32, 64 channels
        self.stage1 = self._make_layer(64, 64, num_blocks=2, stride=1, norm_layer=norm_layer, activation=activation, dropout=dropout, attention_dropout=attention_dropout, stochastic_depth_prob=stochastic_depth_prob)
        
        # Stage 2: 16x16, 128 channels
        self.stage2 = self._make_layer(64, 128, num_blocks=2, stride=2, norm_layer=norm_layer, activation=activation, dropout=dropout, attention_dropout=attention_dropout, stochastic_depth_prob=stochastic_depth_prob)
        
        # Stage 3: 8x8, 256 channels
        self.stage3 = self._make_layer(128, 256, num_blocks=2, stride=2, norm_layer=norm_layer, activation=activation, dropout=dropout, attention_dropout=attention_dropout, stochastic_depth_prob=stochastic_depth_prob)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, norm_layer, activation, dropout, attention_dropout, stochastic_depth_prob):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout, use_se=True, stochastic_depth_prob=stochastic_depth_prob, norm_layer=norm_layer, activation=activation))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout, use_se=True, stochastic_depth_prob=stochastic_depth_prob, norm_layer=norm_layer, activation=activation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}