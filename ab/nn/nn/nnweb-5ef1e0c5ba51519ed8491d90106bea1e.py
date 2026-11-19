import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, drop_p=0.0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop_p = drop_p

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.drop_p > 0.0:
            out = nn.functional.dropout(out, p=self.drop_p, training=self.training)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
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
        input_channels = in_shape[1]
        num_classes = out_shape[0]

        norm_layer = nn.BatchNorm2d
        drop_p = prm['dropout']
        self.attention_dropout = prm['attention_dropout']
        self.stochastic_depth_prob = prm['stochastic_depth_prob']

        # Stem: 3x3 conv, 64 channels, stride 1, BatchNorm, ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        # Four stages of two BasicBlocks with widths 64, 128, 256, 512
        # Stride 1 in stage 1, stride 2 in first block of stages 2, 3, 4
        self.layer1 = self._make_layer(64, 64, 2, stride=1, norm_layer=norm_layer, drop_p=drop_p)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, norm_layer=norm_layer, drop_p=drop_p)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, norm_layer=norm_layer, drop_p=drop_p)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, norm_layer=norm_layer, drop_p=drop_p)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_planes, planes, num_blocks, stride, norm_layer, drop_p):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, norm_layer=norm_layer, drop_p=drop_p))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, norm_layer=norm_layer, drop_p=drop_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        
        # Stage 1
        x = self.layer1(x)
        if self.stochastic_depth_prob > 0.0 and self.training:
            x = self._stochastic_depth(x, self.stochastic_depth_prob)

        # Stage 2
        x = self.layer2(x)
        if self.stochastic_depth_prob > 0.0 and self.training:
            x = self._stochastic_depth(x, self.stochastic_depth_prob)

        # Stage 3
        x = self.layer3(x)
        if self.stochastic_depth_prob > 0.0 and self.training:
            x = self._stochastic_depth(x, self.stochastic_depth_prob)

        # Stage 4
        x = self.layer4(x)
        if self.stochastic_depth_prob > 0.0 and self.training:
            x = self._stochastic_depth(x, self.stochastic_depth_prob)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply attention dropout if specified (simulating attention mechanism dropout on features)
        if self.attention_dropout > 0.0 and self.training:
            x = nn.functional.dropout(x, p=self.attention_dropout, training=self.training)

        x = self.classifier(x)
        return x

    def _stochastic_depth(self, x, drop_prob):
        if drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - drop_prob
        batch_size = x.size(0)
        mask = torch.bernoulli(torch.full((batch_size, 1, 1, 1), keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}