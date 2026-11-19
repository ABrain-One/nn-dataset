from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

class _SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 8), ch),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.downsample = None
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion * out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_ch)
            )
        
        self.se = _SEBlock(self.expansion * out_ch)
        self.drop = nn.Dropout(drop_rate)

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
        
        out = self.se(out)
        out = self.drop(out)
        
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
        
        # ResNet-18 architecture adapted for CIFAR
        # CIFAR inputs are 32x32. Standard ResNet uses 7x7 stride 2. 
        # We use 3x3 stride 1 for the first layer to preserve resolution better, 
        # then stride 2 in the first block.
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Blocks
        # Layer 1: 32x32 -> 32x32 (stride 1)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, drop_rate=dropout)
        
        # Layer 2: 32x32 -> 16x16 (stride 2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, drop_rate=dropout)
        
        # Layer 3: 16x16 -> 8x8 (stride 2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, drop_rate=dropout)
        
        # Layer 4: 8x8 -> 4x4 (stride 2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, drop_rate=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        self._init_weights()

    def _make_layer(self, in_ch: int, out_ch: int, num_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(_BasicBlock(in_ch, out_ch, stride, drop_rate))
            in_ch = out_ch * _BasicBlock.expansion
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
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}