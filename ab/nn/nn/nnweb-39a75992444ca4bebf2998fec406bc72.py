import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()  # Non-inplace to avoid autograd issues

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()  # Non-inplace

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity  # Non-inplace addition
        return self.relu(out)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

        # Initial stem: 3x3 conv, stride 1, no maxpool (better for small images like CIFAR)
        self.stem = ConvBlock(channel_number, 64, kernel_size=3, stride=1, padding=1)

        # Stage 1: 64 channels, spatial size preserved
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1),
        )

        # Stage 2: 128 channels, stride 2
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1),
        )

        # Stage 3: 256 channels, stride 2
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
        )

        # Stage 4: 512 channels, stride 2
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512, class_number)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_layer(x)
        return self.fc(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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