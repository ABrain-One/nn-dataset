import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer(32, 32, 2)
        self.layer2 = self.make_layer(32, 64, 2)
        self.layer3 = self.make_layer(64, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.criteria = nn.CrossEntropyLoss()
        self._set_optimizer(prm)

    def make_layer(self, in_channels, out_channels, blocks):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _set_optimizer(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self._set_optimizer(prm)

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

params = 500000
import numpy as np
import math
def compute_out_channels(in_channels, kernel_size, stride, padding, out_channels=None):
    if out_channels is None:
        out_channels = in_channels
    dilated_kernel_size = kernel_size + (kernel_size - 1) * (stride - 1)
    dilated_padding = padding + (kernel_size - 1) * (stride - 1)
    out_feature_map_size = math.floor(
        (in_channels + 2 * dilated_padding - dilated_kernel_size) / stride + 1)
    out_channels_per_group = out_channels / in_channels
    out_channels = out_channels_per_group * in_channels
    return out_channels, out_feature_map_size

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expansion):
        super(Block, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, in_channels * expansion, kernel_size=1, stride=1, padding=0)
        out_channels, out_feature_map_size = compute_out_channels(
            in_channels * expansion, kernel_size, stride, padding, out_channels)
        self.conv2 = ConvBNReLU(in_channels * expansion, out_channels, kernel_size, stride, padding)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv3 = nn.Sequential()
            self.bn3 = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.conv3.conv.in_channels != self.conv3.conv.out_channels:
            residual = self.conv3(residual)
            residual = self.bn3(residual)
        x += residual
        return x

    def supported_hyperparameters():
        return {'lr', 'momentum'}

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )
        self._last_result = None

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
            self._last_result = outputs.argmax(dim=1)

    def __repr__(self):
        return f'Block(in_channels={self.conv1.conv.in_channels}, out_channels={self.conv2.conv.out_channels}, kernel_size={self.conv2.conv.kernel_size}, stride={self.conv2.conv.stride}, padding={self.conv2.conv.padding})'

    def get_penultimate(self):
        return self._last_result

    def get_penultimate_loss(self, target):
        if self._last_result is None:
            raise ValueError('Call learn() first to produce penultimate activations.')
        return self.criteria(self._last_result, target)

    def get_penultimate_grad(self):
        if self._last_result is None:
            raise ValueError('Call learn() first to produce penultimate activations.')
        return self._last_result.grad

    def get_last_weight(self):
        for name, param in self.named_parameters():
            if 'conv3.conv.weight' in name:
                return param
        raise ValueError('No such parameter: "conv3.conv.weight"')

    def get_last_bias(self):
        for name, param in self.named_parameters():
            if 'conv3.conv.bias' in name:
                return param
        raise ValueError('No such parameter: "conv3.conv.bias"')