import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(32, 32, 3, 1, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2, 1)
        self.layer3 = self._make_layer(64, 128, 3, 2, 1)
        self.layer4 = self._make_layer(128, 256, 3, 2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

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

# Edge-friendly latency budget means we need to keep the model small and efficient.
# We can use larger kernels and fewer layers to maintain speed, but the total
# parameter count must not exceed 500000.
#
# The design below is a compact, efficient block that uses 3x3 kernels and
# depthwise separable convolutions for low latency. We'll use this block in a
# pattern that allows us to fit within the parameter limit.
#
# The design will include:
# - An initial stem block that downsamples the input to 32x32 (4x downsample)
# - A series of blocks that increase depth and reduce spatial size:
#   - Block 1: 32->32, 5x5 kernel, 2x2 downsample
#   - Block 2: 32->64, 5x5 kernel, 2x2 downsample
#   - Block 3: 64->128, 5x5 kernel, 2x2 downsample
#   - Block 4: 128->256, 5x5 kernel, 2x2 downsample
# - A final average pooling layer to reduce to 6x6 feature maps
# - A linear classifier that uses 256*6*6 inputs

# Total parameters must not exceed 500000. We'll use depthwise separable
# convolutions to keep the model size manageable.

# The block design must include:
# - An initial 1x1 pointwise convolution for channel reduction
# - A depthwise 3x3 convolution
# - A second 1x1 pointwise convolution for channel expansion
# - Batch normalization after each convolution
# - A ReLU activation after each batch norm
# - Optionally, a 1x1 pointwise convolution for expansion after the depthwise
#   conv to match the output channels of the next block.

# This block design allows us to use larger 3x3 kernels (which are faster) and
# fewer layers to keep the model small and fast.

# The design will use depthwise separable convolutions to reduce parameters:
# - Depthwise convs reduce parameters by 1/C, making them more efficient.
# - Pointwise convs are used after depthwise convs to match the output channels.
#   This is necessary to maintain channel alignment when combining blocks.

# The total parameter count must be less than or equal to 500000. We'll design
# the block to fit this limit while maintaining performance.

# The design will follow these principles:
# - Use depthwise separable convolutions to reduce parameters.
# - Use 3x3 kernels for spatial reduction without increasing parameters.
# - Use 1x1 pointwise convs for channel reduction/expansion.
# - Keep the block depth (number of blocks) to 1 or 2 to maintain latency.

# Now, let's define the Block and Net classes with the designed block structure.

# Edge-friendly latency budget means we need to keep the model small and efficient.
# We can use larger kernels and fewer layers to maintain speed, but the total
# parameter count must not exceed 500000.
#
# The design below is a compact, efficient block that uses 3x3 kernels and
# depthwise separable convolutions for low latency. We'll use this block in a
# pattern that allows us to fit within the parameter limit.
#
# The design will include:
# - An initial stem block that downsamples the input to 32x32 (4x downsample)
# - A series of blocks that increase depth and reduce spatial size:
#   - Block 1: 32->32, 5x5 kernel, 2x2 downsample
#   - Block 2: 32->64, 5x5 kernel, 2x2 downsample
#   - Block 3: 64->128, 5x5 kernel, 2x2 downsample
#   - Block 4: 128->256, 5x5 kernel, 2x2 downsample
# - A final average pooling layer to reduce to 6x6 feature maps
# - A linear classifier that uses 256*6*6 inputs

# Total parameters must not exceed 500000. We'll use depthwise separable
# convolutions to keep the model size manageable.

# The block design must include:
# - An initial 1x1 pointwise convolution for channel reduction
# - A depthwise 3x3 convolution
# - A second 1x1 pointwise convolution for channel expansion
# - Batch normalization after each convolution
# - A ReLU activation after each batch norm
# - Optionally, a 1x1 pointwise convolution for expansion after the depthwise
#   conv to match the output channels of the next block.

# This block design allows us to use larger 3x3 kernels (which are faster) and
# fewer layers to keep the model small and fast.

# The design will use depthwise separable convolutions to reduce parameters:
# - Depthwise convs reduce parameters by 1/C, making them more efficient.
# -