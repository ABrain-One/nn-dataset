import torch
import torch.nn as nn
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SimpleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = int(32)
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1), nn.BatchNorm2d(start_chan), nn.ReLU(inplace=True))
        blocks = []
        pools = []
        trans_layers = []
        cur_chan = start_chan
        total_blocks = int(2)
        for i in range(total_blocks):
            blocks.append(SimpleBlock(cur_chan, cur_chan, 0.0))
            pools.append(nn.MaxPool2d(2))
            if i < total_blocks - 1:
                next_chan = int(cur_chan * 2)
                trans_layers.append(nn.Sequential(nn.Conv2d(cur_chan, next_chan, kernel_size=1), nn.BatchNorm2d(next_chan), nn.ReLU(inplace=True)))
                cur_chan = next_chan
            else:
                trans_layers.append(None)
        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.trans_layers = nn.ModuleList([t for t in trans_layers if t is not None])
        self.final_channels = cur_chan
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.final_channels, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.entry(x)
        t_idx = 0
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            x = block(x)
            x = pool(x)
            if i < len(self.trans_layers):
                x = self.trans_layers[t_idx](x)
                t_idx += 1
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999))
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
            self.optimizer.step()