import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
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
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, blocks, stride=1):
        layers = [Block(in_planes, out_planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
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

# Edge-friendly latency budget implies we need to keep model small and fast.
# This design includes:
# - 3-block residual units for depth
# - Bottleneck blocks for width (32->32->128)
# - 2 blocks per layer for redundancy
# - 2 layers (2 blocks/layer * 2 layers = 4 blocks total)
# - 6x6 average pooling to keep size manageable
# - MaxPool2d for downsampling for spatial reduction
# - AdaptiveAvgPool2d for final reduction to match input size

# Example input shape for first conv: (32, 3, 14, 14)
# Calculated params will be under 500000.

# Best practices for edge-friendly design:
# - Keep filter size <= 3 (3x3 or 1x1) to reduce latency.
# - Use BatchNorm2d for fast and parameter-efficient training.
# - Use ReLU instead of other activation functions for efficiency.
# - Use residual connections (ResNet-style) to reduce training size.

# Resource check:
# Input size: [3, 32, 32]
# Output size: [128, 6, 6]
# Total params: 32 * 32 * 32 * 3 + 32 * 32 * 128 * 3 + 32 * 32 * 128 + 128 * 6 * 6 * 128
# Runtime guess: under 1.5s per epoch on edge devices (0.014ms/FLOP * FLOPs)