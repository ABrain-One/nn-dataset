from __future__ import annotations
from typing import Callable
import math
from copy import deepcopy
from random import random, randrange
from packaging import version
import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch import nn, einsum, tensor, Tensor, cat, stack, arange, is_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.nn import Module, ModuleList, ModuleDict
from functools import partial, wraps
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

import torch, torch.nn as nn


class RMSNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma


import torch.nn as nn

def supported_hyperparameters():
    return {'lr','momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        # Stable 2D stem to avoid channel/shape mismatches
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # Example patterns you may use (choose ONE appropriate to the block):
        # layers += [ProvidedBlock(in_ch=32, out_ch=32)]
        # layers += [WinAttnAdapter(dim=32, window_size=(7,7), num_heads=4, block_cls=ProvidedBlock)]

        # Keep under parameter budget and end with a known channel count:
        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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