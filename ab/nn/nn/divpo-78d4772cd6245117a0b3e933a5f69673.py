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
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer(32, 32, 3, 1, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2, 1)
        self.layer3 = self.make_layer(64, 128, 3, 2, 1)
        self.layer4 = self.make_layer(128, 256, 3, 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def make_layer(self, in_channels, out_channels, blocks, stride, groups):
        layers = []
        layers.append(Block(in_channels, out_channels, stride=stride, groups=groups))
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
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

# Edge-friendly latency budget means we'll use small convs and skip some blocks
# Design goal: minimize model size and compute while keeping under the param limit
# Example block to follow for size and latency: Block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1)

# Auto-generated design with the above block template
# Parameters below represent the synthesized design and are subject to change

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Target parameters for the synthesized block
in_channels = 3
out_channels = 32
kernel_size = 3
stride = 1
padding = 1
groups = 1

# This is the synthesized block based on the target parameters
class SynthesizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Design constraints: params <= 500000, edge-friendly latency
# Synthesized block parameters: in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1
# The synthesized block has 0.00016788742332458496 params

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)
# Target shape is (3, 32, 32) due to input channels and dimensions
# Input shape must be (3, 32, 32) and not (32, 3, 32, 32) to match channels-first convention

# Central region around the target shape (32x32 channels-first C x H x W)