from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'dropout_aux'}


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
        num_classes: int = out_shape[0]
        dropout = prm['dropout']
        dropout_aux = prm['dropout_aux']
        in_channels = in_shape[1]

        # Standard CIFAR ResNet-18 layout:
        # Stem: 3x3 conv, 64 channels, stride 1, BN, ReLU (no max-pool)
        # Stages: 2 BasicBlocks each, widths 64, 128, 256, 512
        # Stride 2 in first block of stages 2-4
        # Shortcut: 1x1 conv + BN if stride/width changes
        # Head: Global Avg Pool + Linear

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stages
        # Stage 1: 64 channels, 2 blocks, stride 1
        self.layer1 = self._make_layer(64, 2, stride=1, in_channels=64)
        # Stage 2: 128 channels, 2 blocks, stride 2
        self.layer2 = self._make_layer(128, 2, stride=2, in_channels=64)
        # Stage 3: 256 channels, 2 blocks, stride 2
        self.layer3 = self._make_layer(256, 2, stride=2, in_channels=128)
        # Stage 4: 512 channels, 2 blocks, stride 2
        self.layer4 = self._make_layer(512, 2, stride=2, in_channels=256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Using both dropouts to satisfy constraints
        # Standard ResNet has no dropout, but we must use the params.
        # I will apply `dropout` after the final conv layer (before GAP) and `dropout_aux` after GAP (before FC).
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_aux = nn.Dropout(p=dropout_aux)
        
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, in_channels: int = 64) -> nn.Sequential:
        layers: List[nn.Module] = []
        downsample = None
        if stride != 1 or in_channels != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers.append(BasicBlock(in_channels, planes, stride, downsample))
        in_channels = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply dropout as required by hyperparameters
        x = self.dropout(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_aux(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out