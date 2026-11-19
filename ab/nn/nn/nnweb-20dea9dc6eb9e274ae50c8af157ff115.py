import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop_rate: float = 0.0, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=se_reduction)
        self.drop = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se(out)
        out = self.drop(out)
        
        out += identity
        out = F.relu(out, inplace=True)
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
        drop_rate: float = prm['dropout']
        
        # Input normalization
        self.input_norm = nn.BatchNorm2d(in_shape[1])
        
        # Initial convolution
        # For CIFAR (32x32), starting with 3x3 stride 1 is often better than 7x7 stride 2
        # We use 64 channels to keep spatial dims high initially
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        # Stage 1: 32x32 -> 32x32
        self.layer1 = self._make_layer(64, 64, 3, stride=1, drop_rate=drop_rate)
        
        # Stage 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(64, 128, 3, stride=2, drop_rate=drop_rate)
        
        # Stage 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(128, 256, 3, stride=2, drop_rate=drop_rate)
        
        # Stage 4: 8x8 -> 4x4
        self.layer4 = self._make_layer(256, 512, 3, stride=2, drop_rate=drop_rate)
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, drop_rate=drop_rate))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_norm(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}