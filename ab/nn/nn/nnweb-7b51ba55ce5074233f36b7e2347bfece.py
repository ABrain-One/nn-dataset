import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_p)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        
        # Initialize hyperparameters
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        dropout_p = 0.1  # Fixed reasonable dropout for ResNet
        
        # Build ResNet-18 style architecture
        # Stage 1
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet Blocks
        # Block 1: 64 -> 64 (stride 1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout_p=dropout_p)
        
        # Block 2: 64 -> 128 (stride 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_p=dropout_p)
        
        # Block 3: 128 -> 256 (stride 2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_p=dropout_p)
        
        # Block 4: 256 -> 512 (stride 2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout_p=dropout_p)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, class_number)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_p):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, dropout_p))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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