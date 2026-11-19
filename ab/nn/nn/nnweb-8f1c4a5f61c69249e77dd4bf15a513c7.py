from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


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
        dropout: float = prm['dropout']
        
        # ResNet-9 architecture for CIFAR-10
        # Structure:
        # 1. Initial Conv 3x3, Stride 1, Padding 1, BN, ReLU
        # 2. Block 1: 2 ResBlocks (No stride, preserve size 32x32)
        # 3. Block 2: 2 ResBlocks (Stride 2, downsample to 16x16)
        # 4. Block 3: 2 ResBlocks (Stride 2, downsample to 8x8)
        # 5. Global Average Pooling
        # 6. Dropout
        # 7. FC Layer
        
        # Standard width for ResNet-9 is often 64 base channels.
        # The source mentions "configurable width (e.g., 0.75 scale)".
        # Standard choice for ResNet-9 (depth 9) typically uses [64, 128, 256] or similar.
        # Let's use base width 64.
        width = 64
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        
        # Block 1: 2 blocks, stride 1 (output 32x32)
        self.layer1 = self._make_layer(width, width, num_blocks=2, stride=1)
        
        # Block 2: 2 blocks, stride 2 (output 16x16)
        self.layer2 = self._make_layer(width, width * 2, num_blocks=2, stride=2)
        
        # Block 3: 2 blocks, stride 2 (output 8x8)
        self.layer3 = self._make_layer(width * 2, width * 4, num_blocks=2, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(width * 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out