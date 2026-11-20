from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_p, inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out += self.shortcut(x)
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
        num_classes = out_shape[0]
        dropout = prm['dropout']
        
        # ResNet-9 Configuration
        # Standard ResNet-9 for CIFAR:
        # Stem: Conv 3->32
        # Block 1: 2 blocks of width 32
        # Block 2: 2 blocks of width 64 (stride 2)
        # Block 3: 2 blocks of width 128 (stride 2)
        # Total depth: 1 (stem) + 2 + 2 + 2 = 7 conv layers in blocks + 1 stem = 8?
        # Wait, "ResNet-9" usually refers to the depth.
        # Common CIFAR ResNet-9 definition:
        # Stem (1)
        # Layer 1: 2 blocks (4 convs)
        # Layer 2: 2 blocks (4 convs)
        # Total 9 conv layers?
        # Let's look at standard definitions.
        # ResNet-11: 1 stem + 2*2 blocks (4) + 2*2 blocks (4) = 9? No.
        # ResNet-18 (ImageNet) is 18 layers.
        # CIFAR ResNet-N usually means N conv layers.
        # ResNet-9:
        # Stem: 1 conv
        # Block 1: 2 blocks (each has 2 convs) -> 4 convs
        # Block 2: 2 blocks (each has 2 convs) -> 4 convs
        # Total: 1 + 4 + 4 = 9 conv layers.
        
        # Widths:
        # Block 1: 32 channels
        # Block 2: 64 channels
        # (Some implementations use 128 for the last block in deeper nets, but for 9 it's usually 32, 64).
        # Let's verify standard ResNet-9 CIFAR widths.
        # Often: 32, 64.
        # If it were ResNet-11, it would be 32, 64, 128.
        # ResNet-9 is 32, 64.
        
        self.in_planes = 32

        # Stem
        self.conv1 = nn.Conv2d(in_shape[1], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Block 1: 2 blocks, width 32, stride 1
        self.layer1 = self._make_layer(32, 2, stride=1, drop_p=dropout)
        
        # Block 2: 2 blocks, width 64, stride 2
        self.layer2 = self._make_layer(64, 2, stride=2, drop_p=dropout)

        # Head
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

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

    def _make_layer(self, plane, num_blocks, stride, drop_p):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride, drop_p=drop_p))
            self.in_planes = plane * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = nn.functional.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}