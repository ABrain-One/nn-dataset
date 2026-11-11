import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
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
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32, stride=1),
            Block(32, 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer(Block, 32, 32, 1)
        self.layer2 = self.make_layer(Block, 32, 64, 2, True)
        self.layer3 = self.make_layer(Block, 64, 128, 2, True)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def make_layer(self, block, in_channels, out_channels, blocks, downsample=False):
        layers = []
        layers.append(block(in_channels, out_channels, stride=1 if not downsample else 2))
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

# Edge-friendly max param count: 500000. Target avg_macs <= 125000000.
# Choose layer types according to stability on CPU; no channel or block magic.
# Optimize for fast forward() and low latency: batch norm should be NCHW.
# Resource analysis: in_channels=3, input size about 32x32 (may be smaller).
# Derive the first conv's out_channels from the above params.
# Target total latency for edge-friendly inference <= 30ms.

# Auto-generated layer type choices (could be repeated; choose up to 3 random ones):
# Block = True
# ConvBN = True
# GhostModule = False
# supported_hyperparameters() indicates: ['lr', 'momentum']

# Analysis: max channels per layer (32) x 2 to 3 branch = 64; double for first conv out_channels.
# Choose out_channels in [32, 64, 128] for the first block.
# No channel or channel//2/4/8 multiplies allowed outside the block (first conv out_channels only).
# Block choices (could be repeated): Block, Block, Block, Block

# Derive the first (and only) block's out_channels from in_channels and out_channels list:
# out_channels = [32, 64, 128] (choose one). Then out_channels[0] is 32, not 64 or 128.
# Resulting first conv out_channels must be <= 32 (in or out of the block).
# Block cannot be after GhostModule (if present).

# The above conditions yield out_channels=32 for the block (first conv).
# The final layer must have in_channels <= 32 (NCHW batch norm).
# The first conv after block must have out_channels=32 and in_channels <= 32.

# Now generate the rest of the model to fit the bound and meet latency:
# Use Block (first) x 2 to 3 times with out_channels=32.
# The first block is already defined above.
# The rest of the Net class (forward, train_setup, learn) must be complete and below this comment.

# The auto-generated block and any helper modules will use NCHW batch norm.
# The first block and its first conv must have out_channels=32 and in_channels <= 32.
# The final layer of Net must have in_channels <= 32 (NCHW).

# The following code will not be present:
# - if outside block: any of [AdaptiveAvgPool2d, AvgPool2d, Conv, Linear, MaxPool2d]
# - if inside block: any of [BatchNorm2d, Dropout, PReLU]
# - any of [F.avg_pool2d, F.max_pool2d] (no inplace)
# - any of [GroupNorm, LayerNorm, SyncBatchNorm]
# - any ATen operators not supported by torch.compile

# The model must generate forward() traces up to and including self.classifier; no eval().
# The edge device does not support some ops (and will not pass model.to(device).train()).
# The maximum latency budget is around 30ms.