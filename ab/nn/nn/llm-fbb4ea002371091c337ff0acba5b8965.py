import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch import Tensor

def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay'}

class ChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        out = self.pool(x)
        out = out.view(b, c)
        out = self.fc(out)
        return out.view(b, c, 1, 1)

class CBAMBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.ca = ChannelAttention(out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv(x)
        out = self.ca(out)
        out = torch.mul(out, self.conv1x1(self.bn1x1(out)))
        if x.shape[2:] != out.shape[2:]:
            out = nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        return out + residual

class Net(nn.Module):

    def __init__(self, in_shape: Tuple[int, int], out_shape: Tuple[int], prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.channels = [64, 128, 256, 512]
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(nn.Conv2d(self.in_channels, self.channels[0], 3, padding=1, bias=False), nn.BatchNorm2d(self.channels[0]), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(self.channels[0], self.channels[1], 3, padding=1, bias=False), nn.BatchNorm2d(self.channels[1]), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(self.channels[1], self.channels[2], 3, padding=1, bias=False), nn.BatchNorm2d(self.channels[2]), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(self.channels[2], self.channels[3], 3, padding=1, bias=False), nn.BatchNorm2d(self.channels[3]), nn.ReLU(inplace=True))
        self.attention = CBAMBlock(self.channels[3], self.channels[3], stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[3], self.num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=prm['weight_decay'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.fc(x)