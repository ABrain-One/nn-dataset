import torch
import torch.nn as nn
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SimpleDropPath(nn.Module):

    def __init__(self, drop_prob: float=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], x.shape[2], x.shape[3]), 1 - self.drop_prob, device=x.device))
        return torch.where(mask == 1, x, torch.zeros_like(x))

class ConvBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int=3, dropout_prob: float=0.1):
        super().__init__()
        self.channels = int(channels)
        self.dropout_prob = dropout_prob
        activation_layer = nn.LeakyReLU(inplace=True)
        conv_layer = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels, bias=False), nn.InstanceNorm2d(channels, affine=True), activation_layer)
        self.conv = conv_layer
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        start_chan = int(16)
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1), nn.BatchNorm2d(start_chan), nn.ReLU(inplace=True))
        blocks = []
        pools = []
        trans_layers = []
        cur_chan = start_chan
        total_blocks = int(3)
        for i in range(total_blocks):
            blocks.append(ConvBlock(cur_chan, 3, 0.1))
            pools.append(nn.AvgPool2d(2))
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
        self.fc_dropout = nn.Dropout(p=0.1)
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