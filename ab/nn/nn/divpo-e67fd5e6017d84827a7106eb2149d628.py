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
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, 3, 1, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2, 1)
        self.layer3 = self._make_layer(64, 128, 3, 2, 1)
        self.layer4 = self._make_layer(128, 256, 3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, groups):
        layers = [Block(in_channels, out_channels, stride=stride, groups=groups)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

# Edge-friendly latency budget implies a focus on simplicity and efficiency.
# Design principles:
# 1. Keep the model shallow to optimize for latency.
# 2. Use standard layers to avoid runtime patching.
# 3. Target a MAC count to stay below the limit.

# Design suggestions based on the above constraints:
# - Use a depth of 1-3 blocks per layer.
# - Avoid attention mechanisms to keep complexity low.
# - Stick to standard kernel sizes (3x3).
# - Optimize for parameters to be below the limit.

# Analyze the budget and choose an appropriate block depth (e.g., 1-2 blocks).
# The block should not exceed 3x3, and it should be simple to avoid attention mechanisms.

# Example block structure: Block(in_channels, out_channels, kernel_size=3, stride=1, padding=1).

# Design guidelines:
# - The total parameter count should not exceed 500000.
# - The edge-friendly latency budget should be strictly adhered to.
# - Only standard layers (no GroupNorm, LayerNorm, or any custom layers) are allowed.
# - Use channels-first (C, H, W) ordering for convolutions.

# Derive the design based on the above guidelines and the resource constraints.

# The final parameter count and latency should be below or close to the targets.
# The model should compile to the target MAC count and run on edge devices within the budget.

# Generate the complete PyTorch model adhering to all the above criteria.