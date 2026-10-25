import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, 3, 1, 1),
            Block(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = nn.Sequential(
            Block(32, 32, 3, 1, 1),
            Block(32, 32, 3, 1, 1)
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, 3, 1, 1, groups=32),
            Block(64, 64, 3, 1, 1)
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, 3, 1, 1, groups=64),
            Block(128, 128, 3, 1, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

# Edge-friendly latency budget: tight
# Target max_FLOPs <= 5e6 (CIFAR-10, edge-friendly)
# Derive the architecture based on the given resource constraints.
# The resulting model should have parameters <= 500000 and max_FLOPs <= 5e6.

# Analyze the resource constraints and design the following architecture:
# [Replace with actual architecture design based on constraints]

# Derive the design based on the given resource constraints:
# Param count <= 500000
# Max FLOPs <= 5e6
# Use standard layers only
# Edge-friendly latency budget: tight
# Channels-first C x H x W input shape
# Derive the first conv's input channels from in_shape[1] (C)

# The resulting model should have:
# - 1 or 2 convolution layers (no skip connections, no attention)
# - 1 or 2 residual blocks (no batchnorm, no SE blocks)
# - Max pool or adaptive pool after the first convolution
# - Global average pooling (GAP) instead of fully connected layers
# - Param count <= 500000
# - Max FLOPs <= 5e6
# - Edge-friendly latency with respect to input shape and parameters

# Example residual block:
# def make_block(in_channels, out_channels, stride=1, expansion=1):
#     return nn.Sequential(
#         Block(in_channels, out_channels, 3, stride, 1),
#         Block(out_channels, out_channels, 3, 1, 1)
#     )

# Example of adaptive pooling:
# self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

# The resulting model should have the following output:
# Net(
#   (stem): Sequential(
#     (0): Block(...)
#     (1): Block(...)
#     (2): MaxPool2d(...)
#   )
#   (layer1): Sequential(...)
#   (layer2): Sequential(...)
#   (layer3): Sequential(...)
#   (avgpool): AdaptiveAvgPool2d(...)
#   (classifier): Linear(...)
# )