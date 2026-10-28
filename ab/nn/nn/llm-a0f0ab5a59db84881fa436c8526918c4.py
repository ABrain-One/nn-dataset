import math
from typing import Any, Callable, List, Optional
import torch
from torch import nn
from torchvision.models._api import WeightsEnum

class ConvBlock(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=1, norm_layer: Callable[..., nn.Module]=nn.BatchNorm2d, activation_layer: Callable[..., nn.Module]=nn.ReLU) -> None:
        layers: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), norm_layer(out_channels), activation_layer(inplace=True)]
        super().__init__(*layers)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, norm_layer: Callable[..., nn.Module]=nn.BatchNorm2d, activation_layer: Callable[..., nn.Module]=nn.ReLU) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu1 = activation_layer(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.relu2 = activation_layer(inplace=True)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), norm_layer(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu2(out + residual)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels: List[int] = prm.get('channels', [64, 128, 256, 512])
        self.num_classes: int = out_shape[0]
        if len(self.channels) != 4:
            raise ValueError('channels must be a list of exactly 4 integers')
        self.in_channels = in_shape[1]
        self.conv1 = ConvBlock(self.in_channels, self.channels[0], kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.channels[0], self.channels[1], depth=2, stride=2)
        self.layer2 = self._make_layer(self.channels[1], self.channels[2], depth=2, stride=2)
        self.layer3 = self._make_layer(self.channels[2], self.channels[3], depth=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[3], self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels: int, out_channels: int, depth: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(depth - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data: List[tuple]) -> None:
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()

def supported_hyperparameters() -> set:
    return {'lr', 'momentum', 'channels'}

class WeightsEnumClass:
    pass

def _regnet(block_params: Any, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> Net:
    norm_layer = kwargs.pop('norm_layer', partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = Net(block_params, norm_layer=norm_layer, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model