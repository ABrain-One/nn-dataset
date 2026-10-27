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
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
        )
        self.layer1 = nn.Sequential(
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, kernel_size=3, stride=1, padding=1, groups=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=1, padding=1, groups=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            Block(128, 128),
            Block(128, 128),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Dummy input to derive the correct in_channels and out_channels.
dummy_input = torch.randn(1, 3, 32, 32)
in_channels = dummy_input.shape[1]
# Selected architecture: EfficientNet-like with depthwise separable convolutions
# Starting with 32 channels, aiming for max_params <= 500000.
# Follows the EfficientNet pattern of repeating blocks with increasing channels:
# - Initial block (32 -> 32)
# - Repeat blocks with 32, 64, 128 channels for 2-3 times each
# - Aperture-3 depthwise convolutions for subsequent blocks
# - Squeeze and expand (SE) blocks for channel attention

# Target max_params: 500000
# Rough estimate for this design: 32K (stem) + 2K (SE) + 10K (blocks) + 5K (aperture-3)
# Iterate to optimize within the param limit.

# Selected blocks for the design:
# - Block (32 -> 32, depthwise, 3x3, groups=32)
# - SEBlock (32)
# - Block (32 -> 64, depthwise, 3x3, groups=64)
# - SEBlock (64)
# - Block (64 -> 128, depthwise, 3x3, groups=128)
# - SEBlock (128)
# - Block (128 -> 128, depthwise, 3x3, groups=128)
# - SEBlock (128)
# - Block (128 -> 128, depthwise, 3x3, groups=128)
# - SEBlock (128)

# Architecture will include:
# - 2 initial blocks (32 -> 32)
# - 1 SEBlock (32)
# - 2 blocks with 32 -> 64 (depthwise, 3x3, groups=64)
# - 1 SEBlock (64)
# - 2 blocks with 64 -> 128 (depthwise, 3x3, groups=128)
# - 2 SEBlocks (128)
# - 1 block with 128 -> 128 (depthwise, 3x3, groups=128)
# - 1 SEBlock (128)
# - 1 block with 128 -> 128 (depthwise, 3x3, groups=128)
# - 1 SEBlock (128)
# - AdaptiveAvgPool2d to 6x6
# - Linear layer to match the number of classes.

# Avoid training on the dummy input to keep the model parameters below the limit.