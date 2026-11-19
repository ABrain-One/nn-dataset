from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

_BN_MOMENTUM = 0.001  # Randomly picked integer value

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int, bn_momentum: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)
        return out


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

    _version = 2

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        
        # Standard ResNet-18 configuration for CIFAR-10
        # 4 stages, 2 BasicBlocks each
        # Widths: 64, 128, 256, 512
        # Strides: 1 for stage 1, 2 for first block of stages 2, 3, 4
        
        # Stem: 3x3 conv, 64 channels, stride 1, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 64 -> 64, stride 1
        self.layer1 = self._make_layer(64, 64, 2, stride=1, bn_momentum=_BN_MOMENTUM)
        
        # Stage 2: 64 -> 128, stride 2
        self.layer2 = self._make_layer(64, 128, 2, stride=2, bn_momentum=_BN_MOMENTUM)
        
        # Stage 3: 128 -> 256, stride 2
        self.layer3 = self._make_layer(128, 256, 2, stride=2, bn_momentum=_BN_MOMENTUM)
        
        # Stage 4: 256 -> 512, stride 2
        self.layer4 = self._make_layer(256, 512, 2, stride=2, bn_momentum=_BN_MOMENTUM)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, num_classes)
        )
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int, bn_momentum: float) -> nn.Sequential:
        layers = []
        layers.append(_BasicBlock(in_ch, out_ch, stride, bn_momentum=bn_momentum))
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_ch, out_ch, 1, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _load_from_state_dict(
            self,
            state_dict: Dict,
            prefix: str,
            local_metadata: Dict,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: List[str],
    ) -> None:
        version = local_metadata.get("version", None)
        if version not in [1, 2]:
            raise ValueError(f"version should be set to 1 or 2 instead of {version}")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )