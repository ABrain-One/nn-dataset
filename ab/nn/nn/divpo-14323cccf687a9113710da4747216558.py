import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

# Edge-friendly latency budget implies efficient layer choices and pruning.
# This design includes:
# - 3x residual blocks for depth
# - 1x attention (SEBlock) for parallelism
# - early-reduction for size control
# - efficient 3x3 convolutions for spatial efficiency
# - SEBlock after block1 for attention and feature refinement
# - SEBlock after block2 for further attention and feature refinement
# - 64x64 avg pooling for size reduction
# - 512-unit classifier for capacity control
# - BatchNorm for activation normalization
# - ReLU for non-linearity
# - MaxPool for downsampling
# - AdaptiveAvgPool for size reduction before classification
# - pruning to stay under the parameter budget

# Configuration:
# First layer input channels: 3 (in_shape[1])
# Layers: Conv2d(3->32, 3x3, 1x1 padding), BatchNorm, ReLU, MaxPool
# After 1st block: Conv2d(32->64, 3x3, 1x1 padding), BatchNorm, ReLU, MaxPool
# SEBlock over block1 (64 channels, SE with 64 units)
# After 2nd block: Conv2d(64->128, 3x3, 1x1 padding), BatchNorm, ReLU, MaxPool
# SEBlock over block2 (128 channels, SE with 128 units)
# AdaptiveAvgPool to 6x6
# Classifier: Conv2d(128*6*6->512), BatchNorm, ReLU, Linear(512->512), BatchNorm, ReLU, Linear(512->10)
# Parameter count: <= 500000

# Example input shape: (1, 3, 32, 32) => first conv input channels: 3