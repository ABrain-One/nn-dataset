from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, momentum: float = 0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum)
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
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
        input_channels = in_shape[1]
        momentum = prm['momentum']

        # Stage 1: 3x3 conv, 64 channels, stride 1
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

        # Stage 2: 64 -> 128 channels, stride 2 (downsampling)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, momentum=momentum),
            ResidualBlock(128, 128, stride=1, momentum=momentum)
        )

        # Stage 3: 128 -> 256 channels, stride 2 (downsampling)
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, momentum=momentum),
            ResidualBlock(256, 256, stride=1, momentum=momentum)
        )

        # Stage 4: 256 -> 256 channels, stride 1 (no downsampling, deeper feature stage)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 256, stride=1, momentum=momentum),
            ResidualBlock(256, 256, stride=1, momentum=momentum)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _shufflenetv2(
        weights: Optional[WeightsEnum],
        progress: bool,
        *args: Any,
        **kwargs: Any,
) -> Net:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = Net(*args, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model