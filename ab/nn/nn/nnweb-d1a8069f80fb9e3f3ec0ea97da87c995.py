import math
from typing import List

import torch
from torch import nn, Tensor


class SE(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Basic residual block:
    Conv2d -> Norm -> Activation -> Dropout -> (SE) -> Conv2d -> Norm
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, drop_rate=0.0):
        super().__init__()
        self.use_se = use_se
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Optional Dropout
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Optional SE
        if use_se:
            self.se = SE(out_channels)
        else:
            self.se = nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        if self.drop_rate > 0:
            out = self.dropout(out)
            
        out = self.se(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = nn.ReLU(inplace=True)(out)
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
        in_channels = in_shape[1]
        
        # Hyperparameters
        dropout_p = prm['dropout']
        
        # Architecture parameters derived from description
        # "Custom small residual network"
        # CIFAR input is typically 32x32.
        # Stem: 3x3 Conv, Norm, Act. Usually stride 1 for CIFAR to keep 32x32 for Stage 1.
        # Stage 1: 32x32
        # Stage 2: 16x16 (stride 2)
        # Stage 3: 8x8 (stride 2)
        
        # Widths: "Small" implies modest widths. 
        # Let's use 64, 128, 256 as a standard small ResNet configuration.
        # Depth: Let's use 2 blocks per stage (like ResNet-18 scaled down or similar).
        
        widths = [64, 128, 256]
        depths = [2, 2, 2]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(widths[0], widths[0], depths[0], stride=1, use_se=True, drop_rate=dropout_p)
        self.layer2 = self._make_layer(widths[0], widths[1], depths[1], stride=2, use_se=True, drop_rate=dropout_p)
        self.layer3 = self._make_layer(widths[1], widths[2], depths[2], stride=2, use_se=True, drop_rate=dropout_p)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Linear(widths[2], num_classes)
        
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, depth, stride, use_se, drop_rate):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_se, drop_rate))
        for _ in range(1, depth):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_se, drop_rate))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}