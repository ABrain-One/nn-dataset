import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(32, 32, 3, 1, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2, 1)
        self.layer3 = self.make_layer(64, 128, 3, 2, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride, groups):
        layers = []
        layers.append(Block(in_channels, out_channels, stride=stride, groups=groups))
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

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

# Edge-friendly latency and parameter budget.
# Optimize for minimal inference latency and model size.
# Design based on the following guidelines:
# - Prefer depth (more layers) over width.
# - Use stride, groups, and bottleneck blocks for efficient spatial downsampling.
# - Stick to standard layer choices (no GELU, LayerNorm, SE blocks).
# - Aim for 500,000 or fewer parameters.
# - The provided in_shape adheres to channels-first (C, H, W) convention.

# Auto-generated design constraints:
# - Parameters ≤ 500000
# - Edge-friendly latency (prefer high throughput, low latency)
# - Standard layer only: no pre-norm, GELU, LayerNorm, or attention
# - Channels-first (C, H, W) input format according to in_shape

# Analyze resource restrictions and design around them.
# Optimize for accuracy within the parameter budget.
# Due to the edge-friendly latency constraint, the design may prioritize speed over traditional MLP-style layers.

# Design principles:
# - Downsample aggressively for spatial focus and feature compression.
# - Use bottleneck blocks (3x3->1x1->3x3) for high density/area per feature.
# - Prefer depth (more 3x3 layers) over wide layers for area efficiency.
# - Experiment with group convolution for increased efficiency per unit area.

# Example of a downsample block:
# class DownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, groups=1):
#         super().__init__()
#         self.block = Block(in_channels, out_channels, stride=stride, groups=groups)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         return self.pool(self.block(x))

# The final model should use blocks that downsample at least once to meet the spatial requirements.

# Target accuracy: to be determined, optimized during design
# Runtime and memory should be efficient to support edge deployment.

# Generate and implement a complete PyTorch network definition below.