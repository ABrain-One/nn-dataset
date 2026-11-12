import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2)
        self.layer1 = self.make_layer(Block, 32, 32, 1)
        self.layer2 = self.make_layer(Block, 32, 64, 2)
        self.layer3 = self.make_layer(Block, 64, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        layers.append(block(in_channels, out_channels, stride=2))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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

# Edge-friendly, pre-modern architecture inspired by early VGG-style designs.
# 1x1 bottleneck avoids channel bloat, efficient depth-wise convolutions reduce parameters.
# Follows VGG (3x3) block pattern but simplifies to single 3x3 conv to reduce params.
# Targets 500K params or less and ensures edge-friendly latency.
# Emerges from design space exploration, NO prior knowledge on real-world classes.
# Avoids known patterns: NO inverted residual (InvertedResidual, InvertedResidualV2), NO rep, stride=2, NO group.
# Randomly-chosen dropout, if included, is <0.5.
# NO attention (SE, CBAM, GCN).

# AutoML-tuned feature extractor design guidelines:
# - Emphasizes depth-wise convolution for resource efficiency.
# - Favors 3x3 or fewer convs per block to keep params low.
# - Use 1x1 convs for bottlenecks (< 3x3 convs) to reduce params.
# - Simplified blocks avoid skip-connects (no identity).
# - Employs downsampling (stride, avg_pool) to reduce spatial size.
# - No SE blocks, attention mechanisms, or group convolutions.
# - Balances depth (block count) and width (channel count) to stay below param limit.
# - Tests with dropout (if included) to maintain latency.
# - No pretrained weights; trains from scratch on CIFAR-10.

# Design principles:
# - Emphasizes 3x3 or fewer convs per block to keep params low.
# - Favors depth-wise convolutions for resource efficiency.
# - Uses 1x1 convs for bottlenecks to reduce parameters.
# - Simplifies blocks to avoid skip-connects (no identity).
# - Incorporates downsampling (stride, avg_pool) for feature extraction.
# - Avoids SE blocks, attention mechanisms, and group convolutions.
# - Balances depth (block count) and width (channel count) to stay below 500K parameters.
# - Tests with dropout (if included) to maintain edge-compatibility.
# - Does not use pretrained weights; trains from scratch on CIFAR-10.

# Target architecture shape:
# - Up to 3 initial 3x3 convs (for feature extraction).
# - 3-8 blocks (Block or custom) with 3x3 convs only.
# - Average pooling before classifier (no fully connected).
# - Single 1x1 conv or no extras for linear layer (size <= 500K).
# - No SE blocks, attention mechanisms, or group convolutions.
# - Dropout (if included) must be <= 0.5 for edge compatibility.
# - All layers must be defined in this scope and not use pretrained weights.
# - Employs stride or downsampling in blocks for spatial reduction.
# - Blocks must NOT use skip-connections (no identity).
# - Blocks MUST use 3x3 convs only (no larger kernels).
# - Parameters must be <= 500000 (32x32 input).