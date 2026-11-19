import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, drop_p: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop_p) if drop_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, drop_p: float = 0.0):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, drop_p=drop_p)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, drop_p=drop_p)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.downsample(identity) + out
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
        drop_p: float = prm['dropout']

        # Input: (N, C, H, W)
        # CIFAR-10 is 3x32x32.
        # We use a standard ResNet-like architecture adapted for CIFAR.
        
        # Stem
        # CIFAR images are small (32x32), so we don't use a large stride stem.
        # We use a 3x3 conv with stride 1 to preserve resolution initially.
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Feature extraction
        # We use 4 stages.
        # Stage 1: 64 channels, 32x32
        # Stage 2: 128 channels, 16x16 (stride 2)
        # Stage 3: 256 channels, 8x8 (stride 2)
        # Stage 4: 512 channels, 4x4 (stride 2)
        
        # Number of blocks per stage: [2, 2, 2, 2] is a good balance for CIFAR-10.
        # Deeper networks (e.g. ResNet-32/56) work well, but 1 epoch training favors slightly shallower or efficient deep networks.
        # Let's go with ResNet-18 equivalent depth for CIFAR: 2 blocks per stage.
        
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, drop_p=drop_p)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, drop_p=drop_p)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, drop_p=drop_p)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, drop_p=drop_p)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, drop_p: float):
        layers = []
        # First block handles the stride
        layers.append(ResBlock(in_channels, out_channels, drop_p=drop_p) if stride == 1 else None)
        # We need to handle stride in the first block if it's not 1.
        # My ResBlock currently has stride 1 in both convs.
        # I need to modify ResBlock to accept stride or create a special first block.
        # Let's refactor ResBlock to accept stride for the first conv.
        
        # Actually, let's just rewrite the layer creation to be explicit.
        layers = []
        layers.append(ResBlock(in_channels, out_channels, drop_p=drop_p, stride=stride))
        for i in range(num_blocks - 1):
            layers.append(ResBlock(out_channels, out_channels, drop_p=drop_p, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, drop_p: float = 0.0, stride: int = 1):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, drop_p=drop_p)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, drop_p=drop_p)
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.downsample(identity) + out
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}