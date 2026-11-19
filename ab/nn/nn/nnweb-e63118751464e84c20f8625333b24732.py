import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_p=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(drop_p)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        # Use a fixed dropout rate since it's not a hyperparameter, 
        # but we must use the 'momentum' and 'lr' from prm.
        # We will use a small dropout for regularization.
        drop_p = 0.1

        # ResNet-18 style architecture adapted for CIFAR (3x3 convs instead of 7x7)
        # Input: 32x32 (assumed for CIFAR, but code derives from in_shape if needed for dynamic pooling)
        # We use AdaptiveAvgPool2d at the end to handle any spatial size.
        
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 1: 32x32 -> 32x32
        self.layer1 = self._make_layer(64, 64, 2, stride=1, drop_p=drop_p)
        
        # Layer 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(64, 128, 2, stride=2, drop_p=drop_p)
        
        # Layer 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(128, 256, 2, stride=2, drop_p=drop_p)
        
        # Layer 4: 8x8 -> 4x4
        self.layer4 = self._make_layer(256, 512, 2, stride=2, drop_p=drop_p)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, class_number)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, drop_p):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, drop_p))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, drop_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
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
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()