# Auto-generated single-file for ConvEmbedding
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import math
import collections
from itertools import repeat
from collections import *

# ---- original imports from contributing modules ----
from torch import nn

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- ConvEmbedding (target) ----
class ConvEmbedding(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            img_size: int = 224,
            patch_size: int = 16,
            stride: int = 8,
            padding: int = 0,
    ):
        super(ConvEmbedding, self).__init__()
        padding = padding
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.height = math.floor((self.img_size[0] + 2 * padding - self.patch_size[0]) / stride + 1)
        self.width = math.floor((self.img_size[1] + 2 * padding - self.patch_size[1]) / stride + 1)
        self.grid_size = (self.height, self.width)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size,
            stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            ConvEmbedding(self.in_channels, 32, img_size=self.image_size, patch_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
