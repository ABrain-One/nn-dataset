import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = Block(32, 64, stride=1)
        self.block3 = Block(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = Block(64, 128, stride=1)
        self.block5 = Block(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool3(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Edge-friendly latency budget: tight
# Params <= 500000
# Standard layers only; no pretrained weights
# CIFAR-10 (32x32 RGB, channels-first C x H x W)
# Hyperparameters: lr, momentum

# Design principles:
# 1. Efficient: target edge-friendly latency and params <= 500000
# 2. Single-path: avoid skip/residual connections
# 3. Broad-coverage: handle diverse shapes and sizes
# 4. Simple: one-stage/no-attn; no replication/upsampling
# 5. Fast-train: high initial momentum (0.9+) and small batch

# Analysis:
# - Target latency suggests aggressive optimization (e.g., no SE blocks)
# - Edge-friendly means no CUDA extensions; must use Python and standard PyTorch
# - Params count heavily; must stay below 500000
# - Can replicate patterns from efficient Net-like architectures

# Example patterns (from efficient Net-like architectures):
# - Single-path with no skip/residual connections
# - Depth-wise separable convs (dw, pointwise) instead of regular convs
# - Squeeze-and-excitation (SE) blocks with 1x1 convs
# - Efficient use of filters to keep params low

# This design should focus on creating a single-path, efficient, and broad-coverage CNN with the given constraints.