import math
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
        in_channels: int = in_shape[1]
        dropout: float = prm['dropout']

        # Width multiplier for scaling
        w = 2.0

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, int(32 * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * w)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32 * w), int(64 * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * w)),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 8x8
        self.stage1 = nn.Sequential(
            DownsampleBlock(int(64 * w), int(128 * w), w),
            DownsampleBlock(int(128 * w), int(256 * w), w)
        )

        # Stage 2: 4x4
        self.stage2 = nn.Sequential(
            DownsampleBlock(int(256 * w), int(512 * w), w),
            DownsampleBlock(int(512 * w), int(512 * w), w)
        )

        # Stage 3: 2x2
        self.stage3 = nn.Sequential(
            DownsampleBlock(int(512 * w), int(1024 * w), w),
            DownsampleBlock(int(1024 * w), int(1024 * w), w)
        )

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(int(1024 * w), num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


class DownsampleBlock(nn.Module):
    """
    A ResNet-style block that downsamples spatial dimensions by 2.
    Uses a 3x3 conv with stride 2 for the projection and 3x3 convs for the residual.
    """
    def __init__(self, in_channels: int, out_channels: int, w: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut projection to match dimensions (stride 2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out