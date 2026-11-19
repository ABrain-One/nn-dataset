import math
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _conv3x3(in_channels: int, out_channels: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def _bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels)


def _relu() -> nn.ReLU:
    return nn.ReLU(inplace=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm_layer: Callable[..., nn.Module] = _bn,
                 activation_layer: Callable[..., nn.Module] = _relu,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.act = activation_layer()
        self.conv2 = _conv3x3(out_channels, out_channels, 1)
        self.bn2 = norm_layer(out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                _conv3x3(in_channels, out_channels, stride),
                norm_layer(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.drop(self.act(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.act(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}


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

        in_channels = in_shape[1]
        num_classes = out_shape[0]
        input_size = in_shape[2]

        # ResNet-9: [1,1,1,1] basic blocks, widths [64,128,256,512], stride 2 at stage boundaries
        num_blocks = [1, 1, 1, 1]
        widths = [64, 128, 256, 512]

        norm_layer = _bn
        activation_layer = _relu
        dropout = prm['dropout']
        sd_prob = prm['stochastic_depth_prob']
        attention_dropout = prm['attention_dropout']
        momentum = prm['momentum']

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            activation_layer(),
        )

        self.layer1 = self._make_layer(64, widths[0], num_blocks[0], stride=1,
                                      norm_layer=norm_layer, activation_layer=activation_layer,
                                      dropout=dropout)
        self.layer2 = self._make_layer(widths[0], widths[1], num_blocks[1], stride=2,
                                      norm_layer=norm_layer, activation_layer=activation_layer,
                                      dropout=dropout)
        self.layer3 = self._make_layer(widths[1], widths[2], num_blocks[2], stride=2,
                                      norm_layer=norm_layer, activation_layer=activation_layer,
                                      dropout=dropout)
        self.layer4 = self._make_layer(widths[2], widths[3], num_blocks[3], stride=2,
                                      norm_layer=norm_layer, activation_layer=activation_layer,
                                      dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int,
                    stride: int, norm_layer: Callable[..., nn.Module],
                    activation_layer: Callable[..., nn.Module], dropout: float) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for i, s in enumerate(strides):
            in_c = in_channels if i == 0 else out_channels
            layers.append(BasicBlock(in_c, out_channels, s, norm_layer, activation_layer, dropout))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
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