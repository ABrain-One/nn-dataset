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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self._make_layer(32, 32, 2, groups=1)
        self.layer2 = self._make_layer(32, 64, 2, groups=32)
        self.layer3 = self._make_layer(64, 128, 2, groups=64)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, groups):
        layers = [Block(in_channels, out_channels, groups=groups)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

# Edge-friendly channel and kernel size choices:
# - Keep kernel sizes <= 7 (prefer smaller for latency)
# - Use groups to control MACs and avoid large channels
# - Keep channel counts low (e.g., <= 32)
# - Use stride=1 and padding=1 for convs to keep spatial size
# - Use 1x1 convs to reduce and expand channels before and after blocks

# Design principles:
# - Keep parameter count below 500,000
# - Use standard layers only
# - Use groups to control complexity and reduce parameters
# - Use 1x1 convs to adjust channels before and after blocks
# - Use max pooling to downsample spatially (e.g., after blocks)
# - Use layer normalization (nn.BatchNorm2d) instead of instance norm

# Example layer-by-layer design with channel counts:
# - 32x32 (3 input channels)
# - 32x32 (32 output channels, 1x1 conv)
# - 16x16 (32 -> 16 output channels, 1x1 conv, stride=2)
# - 16x16 (16 output channels)
# - 16x16 (16 output channels)
# - 8x8 (16 -> 8 output channels, 1x1 conv, stride=2)
# - 8x8 (8 output channels)
# - 8x8 (8 output channels)
# - 8x8 (8 output channels)
# - 6x6 (8 -> 6 output channels, 1x1 conv, stride=2)
# - 6x6 (6 output channels)
# - 6x6 (6 output channels)
# - 6x6 (6 output channels)
# - 6x6 (6 output channels)
# - 6x6 (6 output channels)
# - 6x6 (6 output channels)
# - 3x3 (6 -> 3 output channels, 1x1 conv, stride=2)
# - 3x3 (3 output channels)
# - 3x3 (3 output channels)
# - 3x3 (3 output channels)
# - 3x3 (3 output channels)
# - 3x3 (3 output channels)
# - 3x3 (3 output channels)
# - 3x3 (3 output channels)
# - AdaptiveAvgPool2d (6 -> 1 output channel)
# - Linear (3 -> 10 output channels)

# Design guidelines:
# - Start with a 32x32 input and reduce it through layers
# - Use 1x1 convs to adjust channels before and after blocks
# - Downsample spatially with max pooling (e.g., stride=2)
# - Keep the final feature map size as small as possible
# - Finish with an AdaptiveAvgPool2d to reduce to 1x1 (channels)
# - Use a Linear layer to output to the correct number of classes

# Hyperparameters and budget:
# - Use prm['lr'] and prm['momentum'] for the optimizer
# - Keep the model small and simple to stay below 500,000 parameters
# - Optimize for accuracy within the above constraints

# Evaluation metric: accuracy on the CIFAR-10 test set.
# If the model repeats classes, the class repetition count is 1.