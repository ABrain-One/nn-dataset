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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = Block(64, 64)
        self.layer2 = Block(64, 64)
        self.layer3 = Block(64, 128)
        self.layer4 = Block(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer3(x)
        x = self.layer4(x)
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

# Edge-friendly channel reduction
# Input shape will be (batch, channels, height, width)
# Derive the first conv's input channels from in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
# Keep channels below 32 to fit within edge budget

# Calculated parameters:
# Total params <= 500000
# Edge-friendly channel count <= 32
# Edge-friendly latency within the budget (edge-friendly = efficient forward pass on edge-class devices)

# Design principles:
# 1. Keep the maximum channel count below 32 to fit within the edge budget.
# 2. Aim for an average channel count per layer that is less than or equal to 32.
# 3. Prioritize simplicity to maintain edge-friendly latency.
# 4. Use standard layers and skip pre-activation layers (e.g., no SeLU, LeakyReLU, or LayerNorm).
# 5. Use standard layer combinations (e.g., Conv-BN-ReLU, no FPN or attention mechanisms).
# 6. Optimize for edge performance and not overall params or peak channel count.

# Resource calculation:
# Start with the input channel count to avoid hardcoding.
# Calculate the maximum number of channels after each layer, aiming to keep it below 32.
# Use standard layers only and avoid complex blocks to maintain edge performance.
# The final parameter count must be under 500000.

# Design guidelines:
# - Prefer sequential blocks (Block, InvertedResidual, etc.) over combinations.
# - Avoid blocks with SeLU, LeakyReLU, or LayerNorm.
# - Use standard Conv-BN-ReLU and avoid Conv-BN-Activation blocks.
# - Use standard pooling (AvgPool2d, MaxPool2d) and avoid attention mechanisms.
# - Aim for an average channel count per layer that is less than or equal to 32.
# - Keep the maximum channel count below 32 to fit within the edge budget.
# - Optimize for forward pass speed and simplicity on edge devices.
# - The model must be fast-forwardable (no inplace ops, no dropout).
# - The edge-friendly latency budget is strict (no complex blocks or large kernels).