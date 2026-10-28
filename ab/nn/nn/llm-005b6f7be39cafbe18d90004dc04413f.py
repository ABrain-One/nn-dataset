import torch
import torch.nn as nn
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, dropout_prob: float=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, dropout_prob: float=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = int(64)
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1), nn.BatchNorm2d(start_chan), nn.ReLU(inplace=True))
        blocks = []
        cur_chan = start_chan
        total_blocks = int(4)
        channels_list = [64, 128, 256, 512]
        for i in range(total_blocks):
            out_channels = int(channels_list[i])
            blocks.append(BottleneckBlock(cur_chan, out_channels, stride=2 if i == 0 else 1, dropout_prob=0.15))
            cur_chan = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout = nn.Dropout(p=0.15)
        self.fc = nn.Linear(cur_chan, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_dropout(x)
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