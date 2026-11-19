import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_p = drop_p

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.drop_p > 0:
            out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out, inplace=True)
        return out


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

        # Standard ResNet-18 architecture adapted for CIFAR
        # CIFAR images are 32x32. We use stride 1 in the first conv to keep spatial dims high.
        
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 1: 32x32
        self.layer1 = self._make_layer(64, 64, 2, stride=1, drop_p=dropout)
        
        # Stage 2: 16x16
        self.layer2 = self._make_layer(64, 128, 2, stride=2, drop_p=dropout)
        
        # Stage 3: 8x8
        self.layer3 = self._make_layer(128, 256, 2, stride=2, drop_p=dropout)
        
        # Stage 4: 4x4
        self.layer4 = self._make_layer(256, 512, 2, stride=2, drop_p=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, class_number)

    def _make_layer(self, in_channels, out_channels, blocks, stride, drop_p):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, drop_p))
        in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(in_channels, out_channels, 1, drop_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
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