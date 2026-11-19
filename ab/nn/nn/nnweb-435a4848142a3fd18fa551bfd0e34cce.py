import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
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
        dropout_prob = prm['dropout']
        
        # ResNet-18 style architecture adapted for CIFAR-10 (32x32)
        # CIFAR-10 inputs are 32x32, so we don't use a 7x7 stem with stride 2.
        # We use a 3x3 conv with stride 1, followed by maxpool (optional, but standard ResNet has it).
        # For CIFAR, often maxpool is omitted or used carefully. Let's stick to standard ResNet-18 structure 
        # but adjusted for 32x32 input.
        
        # Stem: Conv 3x3, 64 channels, stride 1, pad 1
        # Then 4 stages: 64, 128, 256, 512 channels
        # Strides: 1, 2, 2, 2
        
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 1: 64 channels, 2 blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, dropout_prob=dropout_prob)
        
        # Stage 2: 128 channels, 2 blocks
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout_prob=dropout_prob)
        
        # Stage 3: 256 channels, 2 blocks
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout_prob=dropout_prob)
        
        # Stage 4: 512 channels, 2 blocks
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, dropout_prob=dropout_prob)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, out_shape[0])
        
        self._init_params()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(ResBlock(in_channels, out_channels, strides[0], dropout_prob))
        in_channels = out_channels
        for stride in strides[1:]:
            layers.append(ResBlock(in_channels, out_channels, stride, dropout_prob))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
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