import torch
import torch.nn as nn
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = int(64)
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1), nn.BatchNorm2d(start_chan), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ConvBlock(start_chan, 128), ConvBlock(128, 256)])
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)), nn.AdaptiveAvgPool2d((1, 1))])
        self.final_channels = 256
        self.fc_dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(self.final_channels, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.entry(x)
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)
        x = x.flatten(1)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
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