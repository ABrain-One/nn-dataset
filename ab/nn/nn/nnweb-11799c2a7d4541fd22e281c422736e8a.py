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

        # Standard ResNet-9 configuration for CIFAR-10 (32x32)
        # Width multiplier 1.0 (standard choice)
        width_mult = 1.0
        base_width = int(64 * width_mult)

        # Stem: Conv -> BN -> ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        # ResNet-9 Structure:
        # 3 blocks of 1 residual block each.
        # No downsampling (stride=1) to preserve spatial information for CIFAR.
        self.layer1 = self._make_layer(base_width, base_width, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(base_width, base_width, num_blocks=1, stride=1)
        self.layer3 = self._make_layer(base_width, base_width, num_blocks=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(base_width, num_classes)

        # Use dropout_aux to satisfy hyperparameter usage requirement.
        # In standard ResNet, aux heads are not present. We apply this dropout
        # to the features before the final classification head to ensure the
        # hyperparameter is utilized as required by the framework constraints.
        self.dropout_aux = nn.Dropout(p=dropout_aux)

        self._init_weights()

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(in_planes, planes, stride=stride if i == 0 else 1))
            in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.dropout_aux(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out