import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    """A residual-style block: two 3x3 convs with batch norm and ReLU,
    plus a skip connection (with 1x1 conv projection when needed)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection: identity if shapes match, otherwise 1x1 conv + bn
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
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
        num_classes: int = out_shape[0]
        input_channels: int = in_shape[1]

        # Hyperparameters from prm
        dropout: float = prm['dropout']
        momentum: float = prm['momentum']
        attention_dropout: float = prm['attention_dropout']
        stochastic_depth_prob: float = prm['stochastic_depth_prob']

        # 4 convolution stages reaching up to 256 channels
        # Stage channels: 64, 128, 256, 256
        stage_channels = [64, 128, 256, 256]

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Build 4 stages with residual blocks
        self.stages = nn.ModuleList()
        prev_channels = 64
        for i, channels in enumerate(stage_channels):
            # First block in each stage (except stage 0) downsamples
            stride = 2 if i > 0 else 1
            blocks = []
            blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            # Second block (residual-style skip connection in deeper stages)
            blocks.append(ResidualBlock(channels, channels, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            prev_channels = channels

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev_channels, num_classes),
        )

        # Use attention_dropout and stochastic_depth_prob in a meaningful way:
        # Apply stochastic depth to the second residual block in deeper stages
        self.stochastic_depth = nn.ModuleList()
        for i in range(len(self.stages)):
            if i > 0:  # deeper stages
                self.stochastic_depth.append(nn.Dropout(stochastic_depth_prob))
            else:
                self.stochastic_depth.append(nn.Identity())

        # attention_dropout is used in the classifier dropout as well
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            # Apply stochastic depth to deeper stages
            x = stage[0](x)
            x = self.stochastic_depth[i](stage[1](x))
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.attention_dropout(x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}