import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- ResNet-18 Style Architecture (Optimized for CIFAR-10) ---
# ResNet-18 is the standard baseline for CIFAR-10 (with 3x3 convs instead of 7x7 stem).
# It has ~11M params if using standard widths, but we can keep it under 5M by using
# slightly smaller widths or just standard ResNet-18 which is actually ~11.7M?
# Wait, standard ResNet-18 (ImageNet) is 11.2M.
# CIFAR-10 ResNet-18 (3x3 stem) is typically ~11M as well.
# The limit is 5,000,000 parameters.
# We need a smaller ResNet. ResNet-14 or ResNet-10 might fit.
# Let's calculate params for ResNet-18 CIFAR:
# Stem: 3->64 (1.7k)
# Layer 1: 2 blocks of 64 (2* (64*64*9 + 64*64*9) ~ 1.1M) -> Total ~2.8M
# Layer 2: 2 blocks of 128 (2* (128*128*9 + 64*128*9) ~ 4.5M) -> Total ~7.3M
# Layer 3: 2 blocks of 256 (2* (256*256*9 + 128*256*9) ~ 18M) -> Total ~25M
# So standard ResNet-18 is too big.
#
# Let's use a "Small ResNet" or "ResNet-14" or similar.
# Or a "ShuffleNet" / "MobileNet" style?
# The prompt asks for a "much better model" that "learns fast".
# ResNet-18 is the gold standard. If we can't fit 18, let's fit 14 or 12.
#
# Let's try a ResNet with widths [64, 128, 256] but fewer blocks?
# ResNet-14: 2 blocks per stage.
# Stem: 3->64
# Stage 1: 2 blocks, 64 ch
# Stage 2: 2 blocks, 128 ch
# Stage 3: 2 blocks, 256 ch
#
# Params:
# Stem: 3*64*9 = 1728
# BN: 128
#
# Stage 1 (64->64):
# Block 1 (Identity):
#   Conv1: 64*64*9 = 36864
#   BN: 128
#   Conv2: 64*64*9 = 36864
#   BN: 128
#   Total: ~74k
# Block 2 (Identity): ~74k
# Stage 1 Total: ~150k
#
# Stage 2 (64->128):
# Block 1 (Shortcut):
#   Shortcut: 64*128*1 = 8192
#   Conv1: 64*128*9 = 73728
#   Conv2: 128*128*9 = 147456
#   BNs: ~384
#   Total: ~230k
# Block 2 (Identity):
#   Conv1: 128*128*9 = 147456
#   Conv2: 128*128*9 = 147456
#   Total: ~295k
# Stage 2 Total: ~525k
#
# Stage 3 (128->256):
# Block 1 (Shortcut):
#   Shortcut: 128*256*1 = 32768
#   Conv1: 128*256*9 = 294912
#   Conv2: 256*256*9 = 589824
#   Total: ~920k
# Block 2 (Identity):
#   Conv1: 256*256*9 = 589824
#   Conv2: 256*256*9 = 589824
#   Total: ~1.18M
# Stage 3 Total: ~2.1M
#
# FC: 256*10 = 2560
#
# Total Estimate: 1.7k + 150k + 525k + 2.1M + 2.5k ~ 2.8M.
# This fits comfortably under 5M.
#
# To be safe and "better", we can add a 4th stage or increase width slightly?
# Let's stick to ResNet-14 (2 blocks per stage) with widths [64, 128, 256].
# This is a very strong baseline for CIFAR-10.

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        n_classes = out_shape[0] if out_shape else 10

        # Standard CIFAR-10 ResNet-14 (2 blocks per stage)
        # Widths: 64, 128, 256
        self.in_planes = 64

        # Stem: 3x3 conv, no stride (CIFAR-10 images are 32x32)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Layers: 2 blocks each for 64, 128, 256 channels
        # Stage 1: 64 -> 64 (32x32)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        # Stage 2: 64 -> 128 (16x16)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        # Stage 3: 128 -> 256 (8x8)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)

        self.linear = nn.Linear(self.in_planes, n_classes)

        # Use hyperparameters if available, though architecture is fixed
        # The prompt says "keep using every hyperparameter named in supported_hyperparameters"
        # supported_hyperparameters returns {'lr', 'momentum'}
        # We use them in train_setup.

        self.to(device)

    def _make_layer(self, plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride))
            self.in_planes = plane * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=1e-4)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches: break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()