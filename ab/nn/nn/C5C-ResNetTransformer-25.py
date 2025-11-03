import math
import torch
import torch.nn as nn
from typing import Tuple

class Net(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=640):
        super(Net, self).__init__()
        self.out_channels = hidden_dim

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, hidden_dim)

    def train_setup(self, lr: float, momentum: float) -> None:
        # Initialize optimizer here
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Forward pass with teacher forcing
        output = self.forward(x)
        # Shape assert for output
        assert output.shape == (x.size(0), 1, hidden_dim), "Output shape is incorrect"
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We assume the input is [B, 3, H, W] and we process it to [B, 1, hidden_dim]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.pool(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        x = self.adaptive_avg_pool(x)
        x = self.fc(x.flatten(1)).unsqueeze(1)
        return x

def supported_hyperparameters():
    return {'lr','momentum'}