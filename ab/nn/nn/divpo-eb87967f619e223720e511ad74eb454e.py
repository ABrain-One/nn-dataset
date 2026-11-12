import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer(Block, 32, 32, 1)
        self.layer2 = self.make_layer(Block, 32, 64, 2, True)
        self.layer3 = self.make_layer(Block, 64, 128, 2, True)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def make_layer(self, block, in_channels, out_channels, blocks, downsample=False):
        layers = []
        if downsample:
            layers.append(block(in_channels, out_channels))
        else:
            layers.append(block(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
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

# Edge-friendly latency budget (+-10%).
# Design principle: reduce feature maps to a minimum (preferably ≤ 16x16) to avoid post-processing scaling.
# Aim for ≤ 500000 parameters, with early pruning on large blocks to stay under.
# Standard layers only, with no pretrained weights.
# Target a throughput of 10-20 top-1 for the final model.

# Auto-generated design criteria:
# - Hyperparameters: lr=0.01, momentum=0.9
# - Resource limit: params ≤ 500000
# - Edge-friendly latency budget: ±10%
# - Standard layers only
# - Prefer ≤ 16x16 feature maps
# - No pretrained weights
# - Aim for 10-20 top-1 throughput

# Proposed block size (width): 32
# Proposed parameter count hint: <= 500000
# Edge-friendly latency target: 0.182s (±10%)

# Analyze the dataset stats (if available) to inform block and channel choices.
# If available, consider min/max width, height, and channels in training and test sets.
# The final block should be ≤ 32 in width to fit within the latency target.

# Example analysis (fake; replace with dataset analysis):
# - Training set stats: mean width=112.4, height=112.4, max channels=192
# - Test set stats: mean width=110.3, height=110.3, max channels=192
# - Common block widths: [32, 32, 32, 32, 32, 32, 32, 32] (repeatable)
# - Channel counts: [32, 64, 128, 256, 32, 64, 128, 256] (repeatable)

# Design goal: achieve ≤ 16x16 feature maps early in the block to minimize post-processing scaling.
# The largest block should be 32x32 or less in width to meet this goal.
# The total parameter count should be ≤ 500000, with early pruning on large blocks to stay under.

# The script will generate a unique structure based on the above criteria.
# Do NOT copy identical blocks from the training set; instead, create new blocks with different widths and channel counts.
#######################################################################################################################