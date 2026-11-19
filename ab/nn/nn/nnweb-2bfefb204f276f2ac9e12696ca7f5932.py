import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation, MLP


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = nn.ReLU(inplace=True)(out)

        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'patch_size'}


def get_divisors(n, res=None):
    res = res or []
    i = 1
    while i <= n:
        if n % i == 0:
            res.append(i)
        i = i + 1
    return res


def get_closest_split(n, close_to):
    all_divisors = get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if close_to < val:
            if ix == 0: return val
            if (val - close_to) > (close_to - all_divisors[ix - 1]):
                return all_divisors[ix - 1]
            return val


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
        image_size: int = in_shape[2]
        
        # Hyperparameters from prm
        dropout = prm['dropout']
        # We must use attention_dropout and patch_size even though they are not strictly needed for a standard CNN.
        # We will use attention_dropout as the dropout rate for the final classification head.
        # We will use patch_size to determine the initial convolution stride, ensuring it divides the image size.
        # Standard CNNs often use stride 1, but to strictly "use" the hyperparameter, we derive the initial stride from it.
        # However, for CIFAR-10 (32x32), a patch_size of 0.1 would be ~3, which is a valid stride.
        # If patch_size leads to a stride that doesn't divide the image, we fall back to 1.
        target_patch = int(image_size * prm['patch_size'])
        if target_patch > 0 and image_size % target_patch == 0:
            initial_stride = target_patch
        else:
            initial_stride = 1
            
        num_classes: int = out_shape[0]

        # Architecture: 4 convolution stages up to 256 channels
        # Stage 1: 3 -> 64
        # Stage 2: 64 -> 128 (downsample)
        # Stage 3: 128 -> 256 (downsample)
        # Stage 4: 256 -> 256 (downsample)
        
        # Using ResidualBlocks as per "residual-style skip connection"
        # We'll use 2 blocks per stage for depth, or just 1 if simple. 
        # "4 convolution stages" usually implies a sequence. 
        # Let's do:
        # Stem: Conv 3->64, BN, ReLU
        # Stage 1: ResBlock 64->64
        # Stage 2: ResBlock 64->128 (stride 2)
        # Stage 3: ResBlock 128->256 (stride 2)
        # Stage 4: ResBlock 256->256 (stride 2)
        
        # To ensure we use 'patch_size' meaningfully, we set the stem stride to initial_stride.
        # If initial_stride is 1, it's standard. If it's larger, it downsamples early.
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=initial_stride, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ResidualBlock(64, 64, stride=1, dropout=dropout)
        
        self.stage2 = ResidualBlock(64, 128, stride=2, dropout=dropout)
        
        self.stage3 = ResidualBlock(128, 256, stride=2, dropout=dropout)
        
        self.stage4 = ResidualBlock(256, 256, stride=2, dropout=dropout)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Head
        # Use attention_dropout for the head dropout to satisfy the constraint
        self.head = nn.Sequential(
            nn.Dropout(p=prm['attention_dropout']),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x