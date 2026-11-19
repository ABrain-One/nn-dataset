from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Basic residual block with 3x3 convs, BatchNorm, ReLU, optional Dropout2d,
    optional SE, and optional Stochastic Depth.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_se: bool = True, use_dropout: bool = False,
                 use_stochastic_depth: bool = False, drop_prob: float = 0.0):
        super().__init__()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_prob = drop_prob

        # First 3x3 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        
        # Second 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        # Optional Dropout2d
        self.dropout = nn.Dropout2d(p=drop_prob) if use_dropout else nn.Identity()

        # Optional SE
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.se(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        if self.use_stochastic_depth:
            if self.training:
                # Stochastic depth
                if torch.rand(1).item() < self.drop_prob:
                    out = torch.zeros_like(out)
            # else: out = out (identity)

        out = out + identity
        out = self.relu2(out)
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

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        
        # Input channels
        in_channels = in_shape[1]
        
        # Stem: 3x3 Conv2d, norm, activation
        # For CIFAR-10 (32x32), we usually don't use stride in the stem to keep 32x32 for Stage 1.
        # The description says "Stage 1: residual blocks at 32x32".
        # So Stem should output 32x32.
        # Conv 3x3, stride 1, padding 1 -> 32x32.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        # Stage 1: 32x32
        # Standard ResNet-18 style: 2 blocks.
        # Weights: 64, 64
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, use_se=True, use_dropout=True, drop_prob=dropout),
            ResidualBlock(64, 64, stride=1, use_se=True, use_dropout=True, drop_prob=dropout)
        )

        # Stage 2: 16x16
        # Downsample from 32x32 to 16x16.
        # Weights: 128, 128
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True, use_dropout=True, drop_prob=dropout),
            ResidualBlock(128, 128, stride=1, use_se=True, use_dropout=True, drop_prob=dropout)
        )

        # Stage 3: 8x8
        # Downsample from 16x16 to 8x8.
        # Weights: 256, 256
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True, use_dropout=True, drop_prob=dropout),
            ResidualBlock(256, 256, stride=1, use_se=True, use_dropout=True, drop_prob=dropout)
        )

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x