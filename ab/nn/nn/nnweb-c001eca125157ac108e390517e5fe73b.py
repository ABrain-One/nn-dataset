from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, bn_momentum: float = 0.001) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=bn_momentum)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_layer(
        block,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int,
        bn_momentum: float
) -> nn.Sequential:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    layers.append(block(in_ch, out_ch, strides[0], bn_momentum=bn_momentum))
    in_ch = out_ch
    for stride in strides[1:]:
        layers.append(block(in_ch, out_ch, stride, bn_momentum=bn_momentum))
    return nn.Sequential(*layers)


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
        
        # Use a small momentum for BatchNorm to adapt quickly to the distribution
        # of the small CIFAR images, especially in the first epoch.
        bn_momentum = 0.001

        # Standard ResNet-18 architecture adapted for CIFAR-10 (32x32 input)
        # We use stride 1 in the first conv to preserve spatial dimensions initially,
        # and stride 2 in the subsequent layers to downsample.
        # This is the standard "ResNet for CIFAR" setup (no 7x7 stride-2 stem).
        
        self.in_planes = 64
        
        # Stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(in_shape[1], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet Blocks
        # Block 1: 64 channels, 32x32
        self.layer1 = self._make_layer(_BasicBlock, self.in_planes, 64, num_blocks=2, stride=1, bn_momentum=bn_momentum)
        
        # Block 2: 128 channels, 16x16
        self.layer2 = self._make_layer(_BasicBlock, self.in_planes, 128, num_blocks=2, stride=2, bn_momentum=bn_momentum)
        self.in_planes = 128
        
        # Block 3: 256 channels, 8x8
        self.layer3 = self._make_layer(_BasicBlock, self.in_planes, 256, num_blocks=2, stride=2, bn_momentum=bn_momentum)
        self.in_planes = 256
        
        # Block 4: 512 channels, 4x4
        self.layer4 = self._make_layer(_BasicBlock, self.in_planes, 512, num_blocks=2, stride=2, bn_momentum=bn_momentum)
        self.in_planes = 512
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.in_planes, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride, bn_momentum):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(block(in_planes, planes, strides[0], bn_momentum=bn_momentum))
        in_planes = planes
        for stride in strides[1:]:
            layers.append(block(in_planes, planes, stride, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out