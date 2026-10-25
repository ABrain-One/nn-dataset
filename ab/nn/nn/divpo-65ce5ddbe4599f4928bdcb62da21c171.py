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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def _make_layer(self, in_planes, out_planes, blocks, stride, groups):
        layers = [Block(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups)]
        for _ in range(1, blocks):
            layers.append(Block(out_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=groups))
        return nn.Sequential(*layers)

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

# Dummy input to derive the correct in_channels and out_shape
dummy_input = torch.randn(1, 3, 32, 32)
# This will be updated to the actual in_channels value
actual_in_channels = 3
out_shape = (10,)

# Parameters within the budget and constraints
model_params = 500000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Design principles:
# 1. Keep memory consumption low to avoid out-of-memory errors.
# 2. Optimize for latency to run efficiently on edge devices.
# 3. Use standard layers and no pretrained weights.
# 4. Derive block and layer structures to match the parameter count.

# Analysis of the dummy input shape (channels-first C x H x W):
# We'll use this to derive the first conv layer's input channels and structure.

# Since this is the first layer, we'll use the actual in_channels value derived from the dummy input shape.
# The out_channels will be designed to hit the parameter count, then we'll reduce if necessary to stay under the limit.

# Design the block structure to fit within the parameter limit.
# The target number of parameters is 500000.
# We'll now calculate the number of trainable parameters for the blocks we design and iterate to meet the parameter count.

# Since this is the first response and there's no previous design to learn from, we'll have to iterate to find the right structure.
# The iteration will be based on the number of output channels of the blocks we design and the resulting number of parameters.

# The loop will look something like this (pseudo-code):
# for in_channels in range(actual_in_channels, 0, -1):
#     for out_channels in range(32, 0, -1):
#         for blocks in range(2, 0, -1):
#             for groups in range(1, 0, -1):
#                 while calculating the number of parameters for the proposed block:
#                     if parameters meet or exceed the limit:
#                         record the block structure and out_channels
#                         break the inner loop
#                     reduce out_channels or groups for the next iteration

# The actual loop will be executed here and will define the Net class with the chosen block structure, out_channels, and the number of blocks that meet or exceed the parameter limit.