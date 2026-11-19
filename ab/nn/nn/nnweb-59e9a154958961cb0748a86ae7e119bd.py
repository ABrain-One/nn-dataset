from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from functools import partial


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

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
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'norm_eps', 'norm_momentum'}


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
        norm_layer = partial(nn.BatchNorm2d, eps=prm['norm_eps'], momentum=prm['norm_momentum'])
        dropout_p = prm['dropout']

        # CIFAR ResNet-18: stem 3x3 stride 1, no max-pool
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(norm_layer, 64, 2, stride=1)
        self.layer2 = self._make_layer(norm_layer, 128, 2, stride=2)
        self.layer3 = self._make_layer(norm_layer, 256, 2, stride=2)
        self.layer4 = self._make_layer(norm_layer, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, norm_layer, planes, blocks, stride=1):
        downsample = None
        in_channels = 64 if blocks == 2 and stride == 1 else None
        # Determine input channels for the first block
        if stride == 1 and planes == 64:
            in_channels = 64
        elif stride == 2 and planes == 128:
            in_channels = 64
        elif stride == 2 and planes == 256:
            in_channels = 128
        elif stride == 2 and planes == 512:
            in_channels = 256
        else:
            # Fallback: compute from previous layer output
            in_channels = 64

        if stride != 1 or in_channels != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(in_channels, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes * BasicBlock.expansion, planes, 1, None, norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)