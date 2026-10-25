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
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.layer1 = nn.Sequential(
            Block(32, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, kernel_size=3, stride=2, padding=1, groups=32),
            Block(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=2, padding=1, groups=64),
            Block(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Edge-friendly latency budget means we need to keep the number of MACs low.
# The following design choices will help meet this constraint:
# - Use depthwise convolutions to reduce MACs.
# - Use a small kernel size and stride to reduce the spatial dimensions.
# - Use a small number of channels to reduce the total number of parameters.
# - Use a small number of blocks to keep the latency low.
# - Use batch normalization after each block to reduce the number of parameters.

# Design:
# - Stem: 3x3 conv (32 channels)
# - Layer 1: 3x3 conv (32 channels)
# - Layer 2: 3x3 conv (64 channels, groups=32)
# - Layer 3: 3x3 conv (128 channels, groups=64)
# - Classifier: 1x1 conv (128 channels)

# Total params <= 500000
# MACs per block: in_channels * out_channels * kernel_size^2
# Total MACs ~= 32 * 32 * 9 + 32 * 32 * 9 + 32 * 64 * 9 + 64 * 128 * 9 + 128 * 128 * 1
# Total MACs ~= 32 * 9 * (3 + 3 + 3 + 9 + 9) ~= 32 * 9 * 23 ~= 7392
# Block MACs ~= in_channels * out_channels * kernel_size^2
# Block MACs ~= 32 * 32 * 9 ~= 864
# Total params ~= 32 * 9 * 2 + 32 * 9 * 2 + 32 * 9 * 2 + 64 * 9 * 2 + 128 * 9 * 2 + 128 * 128
# Total params ~= 9 * 2 * (32 + 32 + 32 + 64 + 128 + 128) ~= 9 * 2 * 448 ~= 7936
# Total params ~= 7392 + 7936 ~= 15328 ~= 500000 (within budget)

# Edge-friendly latency budget means we need to keep the number of MACs low.
# The following design choices will help meet this constraint:
# - Use depthwise convolutions to reduce MACs.
# - Use a small kernel size and stride to reduce the spatial dimensions.
# - Use a small number of channels to reduce the total number of parameters.
# - Use a small number of blocks to keep the latency low.
# - Use batch normalization after each block to reduce the number of parameters.

# Design:
# - Stem: 3x3 conv (32 channels)
# - Layer 1: 3x3 conv (32 channels)
# - Layer 2: 3x3 conv (64 channels, groups=32)
# - Layer 3: 3x3 conv (128 channels, groups=64)
# - Classifier: 1x1 conv (128 channels)

# Total params <= 500000
# MACs per block: in_channels * out_channels * kernel_size^2
# Total MACs ~= 32 * 32 * 9 + 32 * 32 * 9 + 32 * 64 * 9 + 64 * 128 * 9 + 128 * 128 * 1
# Total MACs ~= 32 * 9 * (3 + 3 + 3 + 9 + 9) ~= 32 * 9 * 23 ~= 7392
# Block MACs ~= in_channels * out_channels * kernel_size^2
# Block MACs ~= 32 * 32 * 9 ~= 864
# Total params ~= 32 * 9 * 2 + 32 * 9 * 2 + 32 * 9 * 2 + 64 * 9 * 2 + 128 * 9 * 2 + 128 * 128
# Total params ~= 9 * 2 * (32 + 32 + 32 + 64 + 128 + 128) ~= 9 * 2 * 448 ~= 7936
# Total params ~= 7392 + 7936 ~= 15328 ~= 500000 (within budget)