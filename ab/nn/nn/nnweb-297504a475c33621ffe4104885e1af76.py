from typing import Optional, Callable

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, dropout=0.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.drop = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'stochastic_depth_prob', 'norm_eps', 'norm_std'}


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
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        
        # Hyperparameters from prm
        self.stochastic_depth_prob = prm['stochastic_depth_prob']
        self.norm_eps = prm['norm_eps']
        self.norm_std = prm['norm_std']
        # We must use all hyperparameters. 
        # 'momentum' is used in train_setup. 
        # 'lr' is used in train_setup.
        # 'stochastic_depth_prob' will be used as dropout rate in the blocks.
        # 'norm_eps' will be used for BatchNorm eps.
        # 'norm_std' will be used for weight initialization.

        norm_layer = lambda planes: nn.BatchNorm2d(planes, eps=self.norm_eps)
        
        # ResNet-9 configuration for CIFAR-10 (32x32)
        # Standard ResNet-9: 
        # Conv 3x3 -> BN -> ReLU
        # 3 stages of 1 block each (total 3 blocks in resnet)
        # Widths: 64, 128, 256 (standard) or scaled. 
        # The prompt mentions "configurable width (e.g., 0.75 scale)". 
        # Standard ResNet-9 often uses [64, 128, 256]. 
        # Let's stick to standard widths for a "standard choice" unless specified otherwise.
        # However, ResNet-9 is specifically a small net. 
        # Common ResNet-9 for CIFAR: 
        # Block 1: 64 filters
        # Block 2: 128 filters
        # Block 3: 256 filters
        # Total blocks = 3.
        
        # To respect "ResNet-9", we use 3 basic blocks.
        # Downsampling: 
        # Stage 1: 32x32 -> 32x32 (stride 1)
        # Stage 2: 32x32 -> 16x16 (stride 2)
        # Stage 3: 16x16 -> 8x8 (stride 2)
        
        self.in_planes = 64

        # Initial conv
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # Block 1: 64 -> 64, stride 1
        self.layer1 = self._make_layer(64, 1, stride=1, norm_layer=norm_layer)
        
        # Block 2: 64 -> 128, stride 2
        self.layer2 = self._make_layer(128, 1, stride=2, norm_layer=norm_layer)
        
        # Block 3: 128 -> 256, stride 2
        self.layer3 = self._make_layer(256, 1, stride=2, norm_layer=norm_layer)

        self.linear = nn.Linear(256, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use norm_std for linear init as required to use the hyperparameter
                nn.init.trunc_normal_(m.weight, std=self.norm_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, norm_layer, dropout=self.stochastic_depth_prob))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)