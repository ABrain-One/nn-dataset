import math
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = F.relu(out, inplace=True)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}


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
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout = prm['dropout']
        attention_dropout = prm['attention_dropout']
        stochastic_depth_prob = prm['stochastic_depth_prob']
        num_classes = out_shape[0]

        # Architecture: 4 convolution stages, up to 256 channels
        # Standard choice: 64 -> 128 -> 256 -> 256
        # Downsampling at stage 2 and 3 (stride 2)
        # Residual blocks in deeper stages (stages 2, 3, 4)

        # Stage 1: Conv 3x3, stride 1, 64 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

        # Stage 2: Residual blocks, 64 -> 128, stride 2
        # "Residual-style skip connection in deeper feature stages"
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=dropout),
            ResidualBlock(128, 128, stride=1, dropout=dropout)
        )

        # Stage 3: Residual blocks, 128 -> 256, stride 2
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout=dropout),
            ResidualBlock(256, 256, stride=1, dropout=dropout)
        )

        # Stage 4: Residual blocks, 256 -> 256, stride 1
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 256, stride=1, dropout=dropout),
            ResidualBlock(256, 256, stride=1, dropout=dropout)
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x