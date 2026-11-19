import re
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param


class ResidualBlock(nn.Module):
    """Basic residual block: two 3x3 convs with batch norm, ReLU, and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_rate = float(drop_rate)

        # Shortcut projection when channels or spatial size change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out += identity
        out = self.relu(out)
        return out


class ConvStage(nn.Module):
    """A convolution stage: a sequence of residual blocks with an initial projection conv."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1, drop_rate: float = 0.0) -> None:
        super().__init__()
        layers = []
        # First block may need to change channels/size
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, drop_rate=drop_rate))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, drop_rate=drop_rate))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


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
        drop_rate: float = prm['dropout']

        # 4 convolution stages reaching up to 256 channels
        # Stage 1: 3 -> 64, no downsampling
        # Stage 2: 64 -> 128, downsample (stride 2)
        # Stage 3: 128 -> 256, downsample (stride 2)
        # Stage 4: 256 -> 256, no downsampling
        # Residual skip connections are used in the deeper stages (2, 3, 4)

        self.stage1 = ConvStage(3, 64, num_blocks=2, stride=1, drop_rate=drop_rate)
        self.stage2 = ConvStage(64, 128, num_blocks=2, stride=2, drop_rate=drop_rate)
        self.stage3 = ConvStage(128, 256, num_blocks=2, stride=2, drop_rate=drop_rate)
        self.stage4 = ConvStage(256, 256, num_blocks=2, stride=1, drop_rate=drop_rate)

        self.classifier = nn.Linear(256, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))


def _build_model(
        num_classes: int,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> Net:
    model = Net((3, 32, 32), (num_classes,), {'dropout': 0.0}, torch.device('cpu'))
    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)
    return model


# Randomized values
growth_rate = 16
block_config = (4, 5, 6, 7)
num_init_features = 64
bn_size = 64
drop_rate = 0.2
num_classes = 1000
in_shape = (3, 32, 32)
out_shape = (1, 1000)
lr = 0.001
momentum = 0.9
num_epochs = 100