import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out += identity
        return F.relu(out, inplace=True)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.num_classes = class_number

        # CIFAR-10 ResNet-20 style architecture (adapted to input size)
        # Depth: 20 (5 blocks per stage, 4 stages)
        # Width: 64, 128, 256, 512

        # Initial convolution
        self.conv1 = ConvBNReLU(channel_number, 64, 3, 1, 1)

        # Stage 1: 64 channels
        self.layer1 = self._make_layer(64, 64, 5, 1, dropout)
        # Stage 2: 128 channels
        self.layer2 = self._make_layer(64, 128, 5, 2, dropout)
        # Stage 3: 256 channels
        self.layer3 = self._make_layer(128, 256, 5, 2, dropout)
        # Stage 4: 512 channels
        self.layer4 = self._make_layer(256, 512, 5, 2, dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, class_number)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'],)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

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