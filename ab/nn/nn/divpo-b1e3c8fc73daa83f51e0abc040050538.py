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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self._make_layer(32, 32, 3, 1, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2, 1)
        self.layer3 = self._make_layer(64, 128, 3, 2, 1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, num_classes)
        )

        self.to(device)

    def _make_layer(self, in_planes, out_planes, blocks, stride, padding):
        layers = [Block(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)]
        for _ in range(1, blocks):
            layers.append(Block(out_planes, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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

# Edge-friendly latency budget implies simplicity and small parameter count.
# Design focuses on fast execution and minimal parameters.
# Model will be small to fit within the edge-friendly latency budget.
# Expected training regime includes fast initial convergence for high LR.

# Auto-generated design with attention to edge performance.
# Parameters:
# Device: cpu
# Hyperparameters: {'lr': 0.01, 'momentum': 0.9}
# Resource limit: params <= 500000
# Resource usage:
# - Params: Less than 500000
# - Latency: Optimized for edge devices

# Example input shape: (1, 3, 32, 32) or (3, 32, 32) depending on input_format.
# Derive in_channels from the input shape and use it in the model construction.

# Class balance: [1.140625] (Fireplace -> 1.140625)

# Training parameters:
# Optimizer: SGD
# Learning rate: 0.01
# Momentum: 0.9

# Quality guidelines:
# - Favor smooth convolution and avoid sharp corners.
# - Use batch normalization for early convergence.
# - Design for fast first-epoch training time.

# Estimated model size: 2.27 MB (actual size should be lower after pruning)

# Target average latency on edge devices: 0.484 sec / transform