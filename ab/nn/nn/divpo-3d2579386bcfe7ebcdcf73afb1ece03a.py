import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
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
            weight_decay=prm.get('weight_decay', 0.0001)
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

# Edge-friendly latency budget: use small layers and avoid complex patterns
# Params <= 500000, no pretrained weights. Follow these guidelines:
#   1. Use standard layer choices (no GroupNorm, NoNorm2, etc.)
#   2. Maximum layer width: channels <= 32
#   3. Use MaxPool2d instead of strides > 2 (for example, 2x2 with stride 2)
#   4. Avoid ResBlocks and skip connections
#   5. Keep the number of layers low (for example, <= 3 per block)

# Derive the design from the following constraints:
#   - Parameters must be <= 500000.
#   - All layers must use standard CNN layers (for example, use nn.Conv2d instead of nn.Conv2dNorm).
#   - The model must be edge-friendly and have the smallest possible spatial dimensions at the output of the first
#     conv layer (for example, avoid layers that would naturally result in dimensions higher than 8x8).

# Analysis of the provided CIFAR-10 dataset characteristics:
#   - Channels: 3 (the channels-first format is (C, H, W), so in_channels will be 3).
#   - Image size: 32x32 (H x W).
#   - Target size after first conv (must be <= 32 to fit in edge-friendly latency budget): 32x32 (H x W).

# Design suggestions:
#   - Use standard CNN layers (no GroupNorm, NoNorm2, etc.).
#   - The smallest spatial dimension after the first conv must be <= 32 (for example, 32 / 2^n <= 32 for some n).
#   - The parameter count must be <= 500000.

# Example of a valid layer combination (the design does not HAVE to use all these layers;
# they are used here to show the product of ideas):
# nn.Sequential(
#     nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
#     nn.BatchNorm2d(32),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, 2)
# )

# The model will be evaluated by accuracy on the CIFAR-10 test set.
# The first implementation or prototype aiming for the lowest test accuracy will have an edge on subsequent rounds.
#[2023-04-10 22:53:34.461471] writing complete entry_point/cifar10_entry.py