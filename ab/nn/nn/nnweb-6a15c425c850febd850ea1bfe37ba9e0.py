import math
import torch
from torch import nn, Tensor
from typing import Tuple


class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions.
    Supports optional Dropout2d and Squeeze-Excitation (SE).
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_dropout: bool = False, dropout_p: float = 0.0,
                 use_se: bool = False, se_ratio: float = 0.25):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout2d(p=dropout_p) if use_dropout else nn.Identity()

        self.use_se = use_se
        if use_se:
            se_dim = max(1, int(out_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(2),
                nn.Linear(out_channels, se_dim),
                nn.ReLU(inplace=True),
                nn.Linear(se_dim, out_channels),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()

        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_dropout:
            out = self.dropout(out)

        if self.use_se:
            out = self.se(out) * out

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
        num_classes = out_shape[0]
        
        # Hyperparameters from prm
        dropout_p = prm.get('dropout', 0.0)
        use_dropout = dropout_p > 0.0
        use_se = False

        # Input shape: (C, H, W)
        in_channels = in_shape[1]
        
        # Stem: 3x3 Conv2d, norm, activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32, 64 channels
        # Stage 2: 16x16, 128 channels
        # Stage 3: 8x8, 256 channels
        
        depths = [2, 2, 2]
        widths = [64, 128, 256]
        strides = [1, 2, 2]

        self.stage1 = self._make_layer(BasicBlock, 64, depths[0], stride=strides[0], in_channels=64, use_dropout=use_dropout, dropout_p=dropout_p, use_se=use_se)
        self.stage2 = self._make_layer(BasicBlock, 128, depths[1], stride=strides[1], in_channels=64, use_dropout=use_dropout, dropout_p=dropout_p, use_se=use_se)
        self.stage3 = self._make_layer(BasicBlock, 256, depths[2], stride=strides[2], in_channels=128, use_dropout=use_dropout, dropout_p=dropout_p, use_se=use_se)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, in_channels=None, use_dropout=False, dropout_p=0.0, use_se=False):
        if in_channels is None:
            in_channels = 64
        layers = []
        layers.append(block(in_channels, planes, stride, use_dropout, dropout_p, use_se))
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, 1, use_dropout, dropout_p, use_se))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum'}