from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor


class ResNetBasicBlock(nn.Module):
    """
    Basic residual block for CIFAR-scale images.
    Uses 3x3 convolutions with stride 1 to avoid aggressive downsampling in early layers.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d, drop_prob: float = 0.0):
        super(ResNetBasicBlock, self).__init__()
        
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        
        # Dropout
        self.drop = nn.Dropout(p=drop_prob)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.drop(out)
        
        out += residual
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
        
        # Hyperparameters
        dropout = prm['dropout']
        norm_eps = prm['norm_eps']
        norm_momentum = prm['norm_momentum']
        
        # Architecture parameters for ResNet-9
        # "ResNet-9" typically implies 9 layers of weight (convolutions).
        # Standard CIFAR ResNet-9 structure:
        # Stem: 1 conv (3x3, stride 1)
        # Block 1: 2 convs (3x3, stride 1) -> 3 total
        # Block 2: 2 convs (3x3, stride 2) -> 5 total
        # Block 3: 2 convs (3x3, stride 2) -> 7 total
        # Head: 1 conv (1x1) -> 8 total? 
        # Wait, usually "ResNet-N" for CIFAR counts the number of 3x3 conv layers in the blocks + stem?
        # Let's look at the standard definition. 
        # ResNet-20: 3 blocks of 2 layers each (6 layers) + 1 stem = 7? No.
        # Common convention: ResNet-N where N is the number of layers.
        # For CIFAR, ResNet-11, 17, 23...
        # ResNet-9 is less standard but often defined as:
        # 1 Stem conv
        # 1 Block with 1 layer? No, residual blocks usually have 2.
        # Let's assume the standard "depth" interpretation where depth = 6*n + 2.
        # If depth=9, that's not a multiple of 6.
        # However, the source "cifar10-resnet9" likely refers to a specific compact model.
        # A common compact ResNet for CIFAR is:
        # Stem: 3x3, S1, 16/32 channels
        # Block 1: 2x 3x3, S1
        # Block 2: 2x 3x3, S2
        # Block 3: 2x 3x3, S2
        # Total 3x3 convs: 1 + 2 + 2 + 2 = 7.
        # Maybe the head is counted? 7 + 1 (1x1) = 8.
        # Or maybe it's a "wide" or "narrow" variant.
        # The prompt mentions "configurable width (e.g., 0.75 scale)".
        # Let's stick to a standard ResNet-9 topology if it exists, or the closest standard compact ResNet.
        # Actually, many implementations of "ResNet-9" for CIFAR use:
        # 1 Stem
        # 1 Block (2 layers)
        # 1 Block (2 layers)
        # 1 Block (2 layers)
        # This is ResNet-11 (1+2+2+2+1 head? No).
        # Let's look at the specific repo name: randyxia07/cifar10-resnet9.
        # A likely structure for a 9-layer model (counting conv layers):
        # Stem (1)
        # Block 1 (2)
        # Block 2 (2)
        # Block 3 (2)
        # Head (1)
        # Total 8?
        # Or maybe:
        # Stem (1)
        # Block 1 (2)
        # Block 2 (2)
        # Block 3 (2)
        # Block 4 (2) -> This is ResNet-17.
        
        # Let's assume the standard CIFAR ResNet structure where "depth" refers to the number of 3x3 conv layers in the blocks plus the stem.
        # If the model is called ResNet-9, it might be:
        # Stem: 1
        # 2 Blocks of 2 layers each? 1 + 4 = 5.
        # 3 Blocks of 2 layers each? 1 + 6 = 7.
        # 4 Blocks of 2 layers each? 1 + 8 = 9.
        # Yes, ResNet-9 likely means 4 basic blocks.
        # Structure:
        # Stem: Conv 3x3, Stride 1, 16 channels (or 32*0.75?)
        # Block 1: 2 layers, Stride 1, 16 channels
        # Block 2: 2 layers, Stride 2, 32 channels
        # Block 3: 2 layers, Stride 2, 64 channels
        # Block 4: 2 layers, Stride 1, 64 channels? Or Stride 2?
        # CIFAR images are 32x32.
        # Stem S1: 32x32
        # Block 1 S1: 32x32
        # Block 2 S2: 16x16
        # Block 3 S2: 8x8
        # Block 4 S1: 8x8
        # This preserves spatial info well.
        
        # Width scale 0.75.
        # Base channels: 16, 32, 64, 128?
        # With 0.75: 12, 24, 48, 96.
        
        # Let's define the standard ResNet-9 (4 blocks) with width_mult 0.75.
        
        width_mult = 0.75
        base_channels = [16, 32, 64, 128]
        channels = [int(c * width_mult) for c in base_channels]
        # Ensure divisibility by 8? Standard practice.
        def make_divisible(v, divisor=8, min_value=None):
            new_v = max(min_value or divisor, int(v + divisor / 2) // divisor * divisor)
            return new_v
        channels = [make_divisible(c) for c in channels]
        
        norm_layer = partial(nn.BatchNorm2d, eps=norm_eps, momentum=norm_momentum)
        
        # Stem
        self.conv1 = nn.Conv2d(in_shape[1], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        # Blocks
        self.layer1 = self._make_layer(channels[0], channels[0], 2, stride=1, norm_layer=norm_layer, drop_prob=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2, norm_layer=norm_layer, drop_prob=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2, norm_layer=norm_layer, drop_prob=dropout)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=1, norm_layer=norm_layer, drop_prob=dropout)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[3], out_shape[0])
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int, norm_layer: Callable[..., nn.Module], drop_prob: float) -> nn.Sequential:
        layers = []
        layers.append(ResNetBasicBlock(in_channels, out_channels, stride, norm_layer, drop_prob))
        for _ in range(1, blocks):
            layers.append(ResNetBasicBlock(out_channels, out_channels, 1, norm_layer, drop_prob))
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
        x = self.classifier(x)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)