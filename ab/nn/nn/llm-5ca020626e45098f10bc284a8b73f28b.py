import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d

class TorchVision(nn.Module):

    def __init__(self, model: str, weights: str='DEFAULT', unwrap: bool=True, truncate: int=1, in_channels: int=3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        if hasattr(torchvision.models, 'get_model'):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if 'aux' in name.lower():
                    continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    if x.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def autocast_ctx(enabled=True):
    return torch.cuda.amp.autocast(enabled=enabled) if torch.cuda.is_available() else contextlib.nullcontext()

def make_scaler(enabled=True):
    return torch.cuda.amp.GradScaler(enabled=enabled) if torch.cuda.is_available() else None
from contextlib import contextmanager

class autocast_ctx:

    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            torch.cuda.amp.autocast()

    def __exit__(self, *args):
        pass

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum', 'use_amp'}

def optimized_conv_block(in_channels, out_channels, stride=1, dropout_prob=0.0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class ChannelGrowthUnit(nn.Module):

    def __init__(self, in_channels, out_channels, num_columns=2, dropout_prob=0.0):
        super().__init__()
        self.num_columns = int(num_columns)
        self.channels = int(out_channels)
        blocks = []
        for i in range(self.num_columns):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = self.channels
            blocks.append(optimized_conv_block(in_ch, self.channels))
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))
        x = torch.stack(outputs, dim=0).mean(dim=0)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_channels = in_channels
        if in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = 'Channel_Optimized_Fractal'
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.2))
        channels = [64, 128, 256, 512]
        self.features = nn.Sequential(optimized_conv_block(in_shape[1], channels[0], stride=1, dropout_prob=dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2), optimized_conv_block(channels[0], channels[1], stride=1, dropout_prob=dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2), optimized_conv_block(channels[1], channels[2], stride=1, dropout_prob=dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2), optimized_conv_block(channels[2], channels[3], stride=1, dropout_prob=dropout_prob), nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_prob), nn.Linear(channels[3], 512), nn.ReLU(inplace=True), nn.Dropout(p=dropout_prob), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, out_shape[0]))

    def _norm4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f'Expected 4D/5D input, got {tuple(x.shape)}')

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        for batch_idx, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()