from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class CIFARResidualNet(nn.Module):
    """
    A custom small residual network for CIFAR-10.
    Layout:
    Stem: 3x3 Conv2d, norm, activation.
    Stage 1: residual blocks at 32x32.
    Stage 2: residual blocks at 16x16.
    Stage 3: residual blocks at 8x8.
    Blocks use 3x3 Conv2d, norm, activation, optional Dropout2d, optional SE/CBAM, and optional stochastic depth.
    Ends with GAP and classifier.
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        
        # Hyperparameters
        dropout_p = prm['dropout']
        momentum = prm['momentum'] # Used in optimizer in train_setup, but must be read here to satisfy LEMUR
        lr = prm['lr'] # Used in optimizer in train_setup, but must be read here to satisfy LEMUR
        
        # Standard choices for "custom small residual network" on CIFAR-10 (32x32 input)
        # Depth: Let's choose a standard small depth, e.g., ResNet-14 or similar (4 blocks per stage? or 3?)
        # The description says "Stage 1... Stage 2... Stage 3".
        # Standard CIFAR ResNets often have 3 stages.
        # Let's use 3 blocks per stage for a "small" net (Total depth ~ 3*3*2 + 1 = 19 layers? Or just 3 blocks per stage).
        # Let's stick to a reasonable "small" size: 3 blocks per stage.
        # Widths: 64, 128, 256 is standard for CIFAR ResNets.
        
        base_width = 64
        blocks_per_stage = 3
        
        # Stem: 3x3 Conv, Norm, Activation. No pooling (CIFAR standard).
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 32x32, width 64
        self.stage1 = self._make_stage(base_width, base_width, blocks_per_stage, dropout_p, stride=1)
        
        # Stage 2: 16x16, width 128
        self.stage2 = self._make_stage(base_width, base_width * 2, blocks_per_stage, dropout_p, stride=2)
        
        # Stage 3: 8x8, width 256
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, blocks_per_stage, dropout_p, stride=2)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 4, self.num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_planes: int, planes: int, num_blocks: int, dropout_p: float, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_planes, planes, stride, dropout_p))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(planes, planes, 1, dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """
    Basic Residual Block for CIFAR.
    Uses 3x3 Convs, BatchNorm, ReLU.
    Includes Dropout2d.
    """
    def __init__(self, in_planes: int, planes: int, stride: int, dropout_p: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Optional Dropout2d
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        
        out += identity
        out = self.relu(out)
        return out


class Net(CIFARResidualNet):
    """
    Wrapper to maintain the exact constructor signature and training methods required by LEMUR.
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__(in_shape, out_shape, prm, device)

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


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}