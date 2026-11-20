import math
from typing import Any, Callable, List, Optional, Sequence
from functools import partial

import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation


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
        norm_eps = prm['norm_eps']
        norm_momentum = prm['norm_momentum']
        
        # Architecture Configuration
        # We use a ResNet-style architecture (ResNet-18/34 hybrid) which is robust and learns fast.
        # For CIFAR-10, we use stride 1 in the first conv and replace maxpools with strided convs.
        
        norm_layer = partial(nn.BatchNorm2d, eps=norm_eps, momentum=norm_momentum)
        
        # BasicBlock definition
        basicblock = BasicBlock
        layer_types = [
            (basicblock, [2, 2, 2, 2]),  # ResNet-18 depth
        ]
        
        # Channel multipliers
        multipliers = [1, 2, 4, 8]
        
        # Input processing
        # CIFAR-10 is 32x32. 
        # Standard ResNet for ImageNet starts with 7x7 stride 2.
        # For CIFAR, we start with 3x3 stride 1.
        # Then we have 4 stages.
        # Stage 1: 32x32 -> 32x32 (stride 1)
        # Stage 2: 32x32 -> 16x16 (stride 2)
        # Stage 3: 16x16 -> 8x8 (stride 2)
        # Stage 4: 8x8 -> 4x4 (stride 2)
        
        # First Conv
        self.conv1 = Conv2dNormActivation(
            in_channels, 
            64, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU
        )
        
        self.inplanes = 64
        self.layer1 = self._make_layer(basicblock, 64, layer_types[0][1][0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(basicblock, 128, layer_types[0][1][1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(basicblock, 256, layer_types[0][1][2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(basicblock, 512, layer_types[0][1][3], stride=2, norm_layer=norm_layer)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )
        
        # Weight Initialization
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride, norm_layer):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'norm_eps', 'norm_momentum'}