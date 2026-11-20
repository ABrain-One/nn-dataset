import math
from typing import List, Optional

import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
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
        dropout: float = prm['dropout']
        
        # Width multiplier for CIFAR-10 (32x32).
        width_mult = 1.0
        norm_layer = nn.BatchNorm2d
        
        # Calculate input channels for first conv
        base_channels = 64
        self.in_channels = int(base_channels * width_mult)
        
        # Stem
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Blocks
        # ResNet18: [2, 2, 2, 2]
        layers = [2, 2, 2, 2]
        
        self.layer1 = self._make_layer(ResidualBlock, int(64 * width_mult), layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(ResidualBlock, int(128 * width_mult), layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(ResidualBlock, int(256 * width_mult), layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(ResidualBlock, int(512 * width_mult), layers[3], stride=2, norm_layer=norm_layer)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(512 * width_mult), num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, norm_layer: Optional[nn.Module] = None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        layers = []
        layers.append(block(self.in_channels, planes, stride, norm_layer))
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, 1, norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}