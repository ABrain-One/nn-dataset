import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout_prob = prm['dropout']
        
        in_channels = in_shape[1]
        
        # Stage 1: 64 channels
        self.block1 = nn.Sequential(
            ResidualBlock(in_channels, 64, stride=1, dropout_prob=dropout_prob),
            ResidualBlock(64, 64, stride=1, dropout_prob=dropout_prob)
        )
        
        # Stage 2: 128 channels, downsample
        self.block2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout_prob=dropout_prob),
            ResidualBlock(128, 128, stride=1, dropout_prob=dropout_prob)
        )
        
        # Stage 3: 256 channels, downsample
        self.block3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout_prob=dropout_prob),
            ResidualBlock(256, 256, stride=1, dropout_prob=dropout_prob)
        )
        
        # Stage 4: 256 channels, no downsample (to keep spatial dims reasonable for 1 epoch)
        self.block4 = nn.Sequential(
            ResidualBlock(256, 256, stride=1, dropout_prob=dropout_prob),
            ResidualBlock(256, 256, stride=1, dropout_prob=dropout_prob)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(256, out_shape[0])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device).to(torch.float32), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()