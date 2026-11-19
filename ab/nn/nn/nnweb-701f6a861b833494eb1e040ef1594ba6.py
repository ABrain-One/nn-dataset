from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class ResBlock(nn.Module):
    """A basic residual block with a 3x3 convolution, batch norm, and ReLU."""
    def __init__(self, channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 architecture adapted for CIFAR-10.
    Uses 3x3 convolutions instead of 7x7 for the initial stem, and removes
    max pooling in favor of stride-2 convolutions to preserve spatial resolution.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        
        # Stem: 3x3 conv with stride 2 (instead of 7x7 stride 2 + maxpool)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResBlock(64 if len(layers)==0 else 64, self._get_dropout())) # Fix: need to pass current channels
        # Correction: _make_layer logic needs to be explicit about input channels
        
        # Let's rewrite _make_layer properly
        layers = []
        layers.append(ResBlock(64, self._get_dropout())) # This is wrong, input channels vary
        
        # Proper implementation:
        layers = []
        # First block might have stride and channel change
        # But standard ResNet18 uses BasicBlock where shortcut handles channel change
        # My ResBlock above assumes same channels. I need to modify ResBlock or use a different approach.
        
        # Let's redefine ResBlock to handle channel changes in the shortcut.
        # Actually, standard ResNet uses a 1x1 conv in shortcut if channels/stride change.
        # Let's stick to a simpler, robust ResNet18 implementation.
        
        # I will redefine the class structure in the main Net to be cleaner.
        pass

    def _get_dropout(self):
        return 0.0 # Placeholder

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        
        # The layers need to be built correctly. Let's discard the broken _make_layer idea
        # and just define the layers in __init__ explicitly or fix _make_layer.
        
        # Since I can't easily edit the class definition flow here without seeing the final code,
        # I will provide the complete corrected file in the output.
        # For this thought block, I'm just verifying the logic.
        
        # Logic for ResNet18:
        # Stem: 3x3, s=1, p=1, 64 ch
        # Layer 1: 2 blocks, 64 ch, s=1
        # Layer 2: 2 blocks, 128 ch, s=2
        # Layer 3: 2 blocks, 256 ch, s=2
        # Layer 4: 2 blocks, 512 ch, s=2
        
        # Shortcut:
        # If stride != 1 or in_ch != out_ch:
        #   Shortcut = Conv2d(in_ch, out_ch, 1x1, stride) + BN
        # Else:
        #   Shortcut = Identity
        
        # My ResBlock above doesn't support shortcut. I need a proper ResBlock.
        
        x = x # Placeholder
        return x


class ProperResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
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
        dropout: float = prm['dropout']
        
        in_channels = in_shape[1]
        
        # ResNet-18 Architecture
        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = []
        layers.append(ProperResBlock(in_planes, planes, stride, dropout))
        for _ in range(1, num_blocks):
            layers.append(ProperResBlock(planes, planes, 1, dropout))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}