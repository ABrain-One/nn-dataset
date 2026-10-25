import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.layer1 = self._make_layer(32, 32, 3, 1, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2, 1)
        self.layer3 = self._make_layer(64, 128, 3, 2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def _make_layer(self, in_planes, out_planes, blocks, stride, groups):
        layers = [Block(in_planes, out_planes, stride=stride, groups=groups)]
        for _ in range(1, blocks):
            layers.append(Block(out_planes, out_planes, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

# Edge-friendly latency budget means we prioritize efficiency over performance.
# Keep the model small, use standard layers, and avoid large kernels.
# Choose a layer size that balances complexity with edge compatibility.

# Design principles:
# - Keep the number of parameters below 500000.
# - Use standard Conv2d + BatchNorm2d blocks with ReLUs.
# - Optimize for edge-friendly latency and resource usage.

# Example of a small, edge-friendly block:
class SmallBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Model architecture (params <= 500000):
# Layer Group  0: Conv2d (32 output features, 3x3, 1 stride, 1 padding)
# Layer Group  1: SmallBlock (32 output features, 3x3, 1 stride, 1 padding)
# Layer Group  2: SmallBlock (32 output features, 3x3, 1 stride, 1 padding)
# Layer Group  3: Conv2d (64 output features, 3x3, 2 stride, 1 padding)
# Layer Group  4: SmallBlock (64 output features, 3x3, 1 stride, 1 padding)
# Layer Group  5: SmallBlock (64 output features, 3x3, 1 stride, 1 padding)
# Layer Group  6: Conv2d (128 output features, 3x3, 2 stride, 1 padding)
# Layer Group  7: SmallBlock (128 output features, 3x3, 1 stride, 1 padding)
# Layer Group  8: SmallBlock (128 output features, 3x3, 1 stride, 1 padding)
# AdaptiveAvgPool2d (6x6 spatial reduction)
# Linear (500000 total weights)

# This model size and parameter budget ensures that the model is both small
# enough to fit on edge devices and complex enough to learn from the data.

# Select an appropriate learning rate and momentum based on the dataset's
# characteristics. Tune the hyperparameters to optimize accuracy on the first
# epoch of training.

# For edge compatibility, choose kernel sizes, padding, and stride that are
# as small as reasonably possible while maintaining spatial resolution.

# The model MUST be defined using the given class and method names. Do NOT
# import additional modules unless they are used for type hints or are part
# of the given class/method signature.

# Resources:
# - PyTorch documentation: https://pytorch.org/docs/stable/index.html
# - Learn about nn.Module: https://pytorch.org/docs/stable/nn.html#torch.nn.Module
# - DL programming patterns: https://pytorch.org/tutorials/intermediate/torch_deep_learning_tricks.html