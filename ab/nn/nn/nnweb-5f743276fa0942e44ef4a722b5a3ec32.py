import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
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

        # Stage 1: 32 channels, stride 2
        self.conv1 = nn.Conv2d(channel_number, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1a = ResBlock(32)
        self.res1b = ResBlock(32)

        # Stage 2: 64 channels, stride 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2a = ResBlock(64)
        self.res2b = ResBlock(64)

        # Stage 3: 128 channels, stride 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.res3a = ResBlock(128)
        self.res3b = ResBlock(128)

        self.classifier = nn.Linear(128, class_number)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1a(x)
        x = self.res1b(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2a(x)
        x = self.res2b(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res3a(x)
        x = self.res3b(x)

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