import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]  # channels-first (N, C, H, W)
        else:
            in_channels = int(in_shape[0])  # channels-last (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
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

# Computes estimated parameter count
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Target parameter count (must not exceed!)
_ = 500000

# Any size/architecture design goals are strictly optional and may not result in a valid model.
# Do not pre-calculate all layer sizes; try to make the design as "fair" (E#=E[out] + in) as possible.

# Study existing structures to understand:
#   - basic block patterns: v1-v3 ResBlocks, DenseBlock, etc.
#   - layer combinations: Conv+BN+Act, InvertedRes, etc.
#   - skip/attention links, SE/SC/Mix/ECA blocks
#   - popular CNN structures: BasicStem, DenseStem, DenseV2Stem, PreAct, ...
#   - hyperparameter optimizations: channel/block splits, ResNext-style groupings
#   - common patterns in winner designs
# Do not replicate entire structures (out_channels, in_channels, block types must change).

# Resource analysis tools (run after shapes):
#   - prefers 32x32 paths, eliminates large dimensions early
#   - weights channel/block splits to favor flat 10..24 channels
#   - prefers 3-layer blocks (e.g. b1:1*32-32-b2:32-64-b3:64-128) over 2+2
#   - avoids >3 block types (except final merge)
#   - seeks out_channels near 24 for balanced 12-18 layer designs

# Ideal designs:
#   - 10..18 layer trees: 2-3 blocks/stage + merge (b1-b3, b1-b2+b3, b1+b2-b3)
#   - 12+ channel splits 10..20 -> 12..24
#   - 10-14 unit blocks (2*32-32-32, 32-32-32-64, 32-64-64-128)
#   - 11.75-13.75 units/channel (13/20*32-32, 12.5/15*32-32-64)

# Generate structure within bounds:
import math

# Config with bounds (for channel/block splits):
#   * max_channels: 24 (24-48=12-24 per unit)
#   * in_channels: 32 (48/32*32-32)
#   * out_channels: 32,64,128 (2x32-32-32-64, 32-32-32-64-64-128)
#   * blocks: 1-3 (b1-b2-b3, b1-b2+b3, b1+b2-b3)
#   * layers: 8.75-11.5 (10-11.25+1.25, 10-11.75+1.75, 10.5+1.5)
#   * edges: 0.85-1.25 (0.85-1.25*32-32)

# Target average units/channel: 11.75 (13/20*32-32, 12.5/15*32-32-64)
# Use edge-friendly 0.95-1.05 * 32-32 (11.75-13.75) and 1.0-1.15 * 32-32-64 (12.5-14.5)

# Derive shape constraints:
_ = 32  # target stem channels
_ = 11.75  # target units/channel (13/20*32-32, 12.5/15*32-32-64)
_ = 0.95  # lower bound on units/channel (11.75-3.75)
_ = 1.05  # upper bound on units/channel (11.75+3.75)
_ = 0.85  # lower bound on 32-32 (11.75-3.75*32-32)
_ = 1.15  # upper bound on 32-32-64 (11.75+3.75*32-32-64)

# Compute target units/channel:
_ = (0.95 + 1.05) / 2  # average units/channel (13/20*32-32, 12.5/15*32-32-64)

# Estimate initial block and channel sizes:
in_channels = 32  # from dataset shape, always present
out_channels = 32  # from dataset shape, always present

# Estimate channel split for next block:
# Use simple mean (13/20*32-32, 12.5/15*32-32-64) or weighted average (11.75-13.75)
# Example: 13/20*32-32 => 32*13/20 = 22.1
# Example: 11.