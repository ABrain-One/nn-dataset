import math
from collections import OrderedDict

import torch
from torch import nn, Tensor


class BasicBlock(nn.Module):
    """Standard CIFAR-style basic residual block (two 3x3 convs, BN+ReLU)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 momentum: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
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
        in_channels: int = in_shape[1]
        momentum: float = prm['momentum']

        # ResNet-9: 4 stages, 2 basic blocks each (2*4 = 8 residual blocks + stem).
        # Widths follow the standard CIFAR ResNet-9 (64 -> 128 -> 256 -> 512).
        # Downsampling (stride 2) is applied only at the start of stages 2-4,
        # so no downsampling happens in the first stage (preserves spatial info).
        widths = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        blocks_per_stage = 2

        # Stem: single 3x3 conv (stride 1) + BN + ReLU.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0], momentum=momentum),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = widths[0]
        for i, (out_ch, stride) in enumerate(zip(widths, strides)):
            blocks = []
            for j in range(blocks_per_stage):
                # First block of a stage may downsample (stride) / change width.
                block_stride = stride if j == 0 else 1
                block_in = in_ch if j == 0 else out_ch
                blocks.append(BasicBlock(block_in, out_ch, block_stride, momentum))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=widths[-1], out_features=num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum'}