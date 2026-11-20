from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout: float, momentum: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=momentum)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=momentum)
        self.dropout = nn.Dropout(p=dropout)
        
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=momentum)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        momentum: float = prm['momentum']

        in_ch = in_shape[1]
        
        # Stage 1: 64 channels
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=momentum)
        
        # Stage 2: 128 channels, stride 2
        self.res2a = _ResBlock(64, 128, stride=2, dropout=dropout, momentum=momentum)
        self.res2b = _ResBlock(128, 128, stride=1, dropout=dropout, momentum=momentum)

        # Stage 3: 256 channels, stride 2
        self.res3a = _ResBlock(128, 256, stride=2, dropout=dropout, momentum=momentum)
        self.res3b = _ResBlock(256, 256, stride=1, dropout=dropout, momentum=momentum)

        # Stage 4: 256 channels, stride 2
        self.res4a = _ResBlock(256, 256, stride=2, dropout=dropout, momentum=momentum)
        self.res4b = _ResBlock(256, 256, stride=1, dropout=dropout, momentum=momentum)

        self.classifier = nn.Linear(256, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.res2a(x)
        x = self.res2b(x)

        x = self.res3a(x)
        x = self.res3b(x)

        x = self.res4a(x)
        x = self.res4b(x)

        x = x.mean([2, 3])
        x = self.classifier(x)
        return x