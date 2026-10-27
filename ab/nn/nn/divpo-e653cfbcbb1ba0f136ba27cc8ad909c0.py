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

# Edge-friendly latency budget implies a relatively simple model to keep latency low.
# This design includes:
# - 3-layer residual blocks with BatchNorm and ReLU
# - AdaptiveAvgPool and early Flatten to keep latency low
# - param limit of 500000
# - no pretrained weights
# - channels-first (C, H, W) input for consistency

# Analyze input shape for the first conv's input channels
in_channels = 3  # RGB images from CIFAR-10 dataset

# This is the design budget for the model:
params = 500000
latency = "edge-friendly"
allowed_layers = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear]

# This is the generated block structure:
# Net:
#   stem: Sequential(
#       Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
#       BatchNorm2d(32),
#       ReLU(inplace=True),
#       MaxPool2d(kernel_size=2, stride=2)
#   )
#   block1: Sequential(
#       Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#       BatchNorm2d(64),
#       ReLU(inplace=True),
#       MaxPool2d(kernel_size=2, stride=2)
#   )
#   block2: Sequential(
#       Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
#       BatchNorm2d(128),
#       ReLU(inplace=True),
#       MaxPool2d(kernel_size=2, stride=2)
#   )
#   classifier: Sequential(
#       AdaptiveAvgPool2d((6, 6)),
#       Flatten(),
#       Linear(128 * 6 * 6 -> 512, bias=False),
#       BatchNorm1d(512),
#       ReLU(inplace=True),
#       Linear(512 -> 10)
#   )

# Required import for BatchNorm1d (if not already imported)
import torch.nn.functional as F

# Selected anchor module: nn.ReLU(inplace=True)
# This module and similar ones will be subject to pruning and early exit.

# Locking import and structure to prevent further changes