import torch.nn as nn
from functools import partial
from typing import Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.normalized_channels = normalized_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, normalized_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_channels, 1, 1))
        else:
            self.weight = None
            self.bias = None
        self.register_buffer('running_mean', torch.zeros(1, normalized_channels, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, normalized_channels, 1, 1))

    def forward(self, x):
        if self.training and self.affine:
            # We are in training mode and affine is True, so we use the affine parameters.
            # Reshape x to (batch, normalized_channels, sequence)
            x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, x.shape[1])
            x = torch.nn.functional.layer_norm(x, (x.shape[2],), self.weight, self.bias, self.eps)
            x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, x.shape[1])
            x = torch.nn.functional.layer_norm(x, (x.shape[2],), self.weight, self.bias, self.eps)
            x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[1]).permute(0, 3, 1, 2).contiguous()
        return x

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            LayerNorm2d(3, affine=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            LayerNorm2d(64, affine=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, num_classes)
        )

    def train_setup(self, hparams):
        # Set up optimizer
        lr = hparams.get('lr', 0.01)
        momentum = hparams.get('momentum', 0.9)
        weight_decay = hparams.get('weight_decay', 0)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def learn(self, batch, **kwargs):
        # This is a training step. We assume batch is (images, labels)
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss

    def forward(self, x, y=None):
        # Shape asserts: x should be a 4D tensor.
        assert x.dim() == 4, "Input must be a 4D tensor"
        # If y is provided, we are in teacher forcing mode.
        if y is not None:
            assert y.dim() == 4, "Teacher forcing target must be a 4D tensor"
        x = self.features(x)
        x = self.classifier(x)
        return x