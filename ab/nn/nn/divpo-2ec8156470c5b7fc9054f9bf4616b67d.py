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
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer4 = Block(128, 128, kernel_size=3, stride=2, padding=1)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Edge-friendly latency budget implies efficient layer choices and minimal preprocessing/postprocessing
# Design principles:
# 1. Keep the maximum path length (number of layers) to 4 or less for edge-friendly latency.
# 2. Use 3x3 or smaller convolutions to reduce parameters and keep latency low.
# 3. Batch norms after each layer to keep the network fat to reduce parameters.
# 4. Avoid seperable convolutions (3x1 or 1x3) to keep complexity under the limit.
# 5. Use 1x1 convolutions sparingly to reduce width and keep under the parameter limit.
# 6. Start with a 32x32 input to maintain resolution for efficient operations.
# 7. Prefer depthwise separable blocks (Block class) to keep parameters low.
# 8. Use stride to reduce spatial dimensions to keep the model fat.

# Design:
# Input shape: [B, 3, 32, 32] (channels-first)
# Pretrained: False
# Parameters: <= 500000
# Edge-friendly latency (slim build) with max path length to 4 or less.
# Blocks: Block (in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1)

# Example simplified block to keep path length under 4:
# Block(in_channels, out_channels) -> Conv2d -> BatchNorm2d -> ReLU

# Resource analysis:
# Given the input shape, the model should produce a feature map of size (N, C, H, W) = (?, 128, 6, 6) after the average pool.
# This should result in a linear input dimension of 128 * 6 * 6 <= 500000 params.

# To adhere to the parameter limit, the design should ensure that the product of out_channels and the spatial product of the feature maps after the last known layer (before the classifier) is less than or equal to 500000.

# Since we're under budget and the input is 32x32, we can afford to have a middle layer with 128 channels and a feature map size of 6x6.

# The target for the last layer before the classifier is out_channels * H * W <= 500000 with H = 6 and out_channels = 128.

# The final layer in the classifier should match the channels of the feature extractor: num_classes == 128.

# This will create a block with in_channels <= 32, out_channels == 128, and kernel_size == 3 to avoid exceeding the parameter budget.

# Please create the complete network with the specified input shape, constraints, and resource limit.