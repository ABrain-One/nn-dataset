import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        
        # Architecture: 4 convolution stages reaching up to 256 channels
        # Stage 1: 3 -> 64
        # Stage 2: 64 -> 128
        # Stage 3: 128 -> 256
        # Stage 4: 256 -> 256
        
        # Initial Conv block
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 1
        self.res1_1 = ResidualBlock(64, 64, stride=1)
        self.res1_2 = ResidualBlock(64, 64, stride=1)
        
        # Stage 2 (Downsample)
        self.res2_1 = ResidualBlock(64, 128, stride=2)
        self.res2_2 = ResidualBlock(128, 128, stride=1)
        
        # Stage 3 (Downsample)
        self.res3_1 = ResidualBlock(128, 256, stride=2)
        self.res3_2 = ResidualBlock(256, 256, stride=1)
        
        # Stage 4
        self.res4_1 = ResidualBlock(256, 256, stride=1)
        self.res4_2 = ResidualBlock(256, 256, stride=1)
        
        self.classifier = nn.Linear(256, class_number)
        
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.res1_1(x)
        x = self.res1_2(x)
        
        x = self.res2_1(x)
        x = self.res2_2(x)
        
        x = self.res3_1(x)
        x = self.res3_2(x)
        
        x = self.res4_1(x)
        x = self.res4_2(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
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