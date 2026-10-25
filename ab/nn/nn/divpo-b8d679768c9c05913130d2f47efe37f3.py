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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(32, 32, 3, 1, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2, 1)
        self.layer3 = self.make_layer(64, 128, 3, 2, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride, padding):
        layers = []
        layers.append(Block(in_channels, out_channels, stride=stride, padding=padding))
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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

# Edge-friendly latency budget implies efficient layer choices and minimal layers.
# Design based on efficient ResNet-34 block and layer patterns:
# - Use Block with stride=1 for residual paths to keep them short.
# - Keep layer counts low (<= 2 blocks per layer) to keep params low.
# - Use 1x1 convolutions for projection skip-connections to reduce channels if needed.

# Design principles:
# - Keep layer count low to meet params constraint.
# - Use 1x1 convolutions to reduce channels if in_channels != out_channels.
# - Use blocks with stride=1 for residual connections to keep them short.
# - Simplify by not using group convolutions or SE blocks.

# Example of a short residual block to keep layer count low:
# Block(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

# Resource calculation:
# - Given in_shape (3, 32, 32), derive in_channels = 3.
# - Desired out_channels: [32, 32, 64, 64, 128, 128] (see make_layer).
# - Total params should not exceed 500000.

# Analysis:
# - Max out_channels per layer to keep params low.
# - Use 1x1 convolutions for projection when in_channels != out_channels.
# - Keep layer count low to meet the edge budget.

# Design guidelines:
# - Focus on making the shortest residual paths.
# - Use 1x1 convolutions for projection when in_channels != out_channels.
# - Keep the total layer count (across all blocks) to 4 or fewer.

# Derive the target out_channels for each layer based on the params limit and the block design:
# - Start with in_channels = 3 and derive out_channels that meet the params constraint.
# - Use the provided in_shape to ensure the input channels match the dataset (in_channels = 3).
# - The edge budget should allow for the shortest possible residual paths.

# Generate the architecture with the derived out_channels and the given in_channels.