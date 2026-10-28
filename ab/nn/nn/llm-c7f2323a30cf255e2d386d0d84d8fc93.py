import torch
import torch.nn as nn
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BottleneckBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, stride: int=1):
        super().__init__()
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.downsample = None
        if stride != 1 or self.channels_in != self.channels_out:
            self.downsample = nn.Sequential(nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.channels_out))
        self.conv1 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels_out)
        self.conv2 = nn.Conv2d(self.channels_out, self.channels_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels_out)
        self.conv3 = nn.Conv2d(self.channels_out, self.channels_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        channels = [64, 128, 256, 512, 512]
        self.entry = nn.Sequential(nn.Conv2d(c_in, channels[0], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(channels[0], channels[1], 1, stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], 2, stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], 2, stride=2)
        self.layer4 = self._make_layer(channels[3], channels[4], 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[4], n_classes)
        self.to(device)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BottleneckBlock(in_channels, out_channels, stride))
        for _ in range(blocks - 1):
            layers.append(BottleneckBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.entry(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=0.0001)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches:
                break
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()