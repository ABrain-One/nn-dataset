from typing import Optional, Callable

import torch
from torch import nn, Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_shortcut: bool = False, norm_layer: Optional[Callable] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.shortcut = nn.Sequential()
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
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
        norm_layer = nn.BatchNorm2d

        # CIFAR-10 specific: 3x3 stem, stride 1, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        # Four stages of two BasicBlocks with widths 64, 128, 256, 512
        # Stage 1: 64 -> 64, stride 1
        self.layer1 = self._make_layer(64, 64, 2, stride=1, norm_layer=norm_layer)
        
        # Stage 2: 64 -> 128, stride 2 (first block)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, norm_layer=norm_layer)
        
        # Stage 3: 128 -> 256, stride 2 (first block)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, norm_layer=norm_layer)
        
        # Stage 4: 256 -> 512, stride 2 (first block)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, norm_layer=norm_layer)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * BasicBlock.expansion, num_classes)
        )

        # Standard initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, norm_layer) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            use_shortcut = (in_channels != out_channels * BasicBlock.expansion) or (s != 1)
            layers.append(BasicBlock(in_channels, out_channels, s, use_shortcut, norm_layer))
            in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x