import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class CIFARResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super(CIFARResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        num_classes = out_shape[0]
        in_channels = in_shape[1]
        dropout = prm['dropout']

        # Stem: 3x3 Conv2d, norm, activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: residual blocks at 32x32
        # Standard choice for CIFAR-10 (32x32 input): 2 blocks
        self.stage1 = nn.Sequential(
            CIFARResidualBlock(64, 64, stride=1, dropout=dropout),
            CIFARResidualBlock(64, 64, stride=1, dropout=dropout)
        )

        # Stage 2: residual blocks at 16x16
        # Downsampling via stride 2 in the first block
        self.stage2 = nn.Sequential(
            CIFARResidualBlock(64, 128, stride=2, dropout=dropout),
            CIFARResidualBlock(128, 128, stride=1, dropout=dropout)
        )

        # Stage 3: residual blocks at 8x8
        # Downsampling via stride 2 in the first block
        self.stage3 = nn.Sequential(
            CIFARResidualBlock(128, 256, stride=2, dropout=dropout),
            CIFARResidualBlock(256, 256, stride=1, dropout=dropout)
        )

        # Ends with GAP and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x