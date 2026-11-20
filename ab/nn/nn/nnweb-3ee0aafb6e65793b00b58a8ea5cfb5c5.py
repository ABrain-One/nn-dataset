import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout_prob = prm['dropout']
        
        # Use a small fraction of the dropout parameter for the classifier head
        # to ensure regularization without killing learning in the final layer
        head_dropout_prob = min(0.5, self.dropout_prob * 0.5) if self.dropout_prob > 0 else 0.0

        # CIFAR-10 is 3x32x32. We use a standard ResNet-style architecture.
        # Input: 3x32x32
        # Stem: 3x3 conv, stride 1, padding 1 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 1: 64 channels, 32x32
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, dropout=self.dropout_prob),
            ResidualBlock(64, 64, stride=1, dropout=self.dropout_prob)
        )
        
        # Block 2: 128 channels, 16x16
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=self.dropout_prob),
            ResidualBlock(128, 128, stride=1, dropout=self.dropout_prob)
        )
        
        # Block 3: 256 channels, 8x8
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout=self.dropout_prob),
            ResidualBlock(256, 256, stride=1, dropout=self.dropout_prob)
        )
        
        # Block 4: 512 channels, 4x4
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, dropout=self.dropout_prob),
            ResidualBlock(512, 512, stride=1, dropout=self.dropout_prob)
        )
        
        # Classification Head
        # Global Average Pooling reduces 512x4x4 to 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout_prob),
            nn.Linear(512, out_shape[0])
        )
        
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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