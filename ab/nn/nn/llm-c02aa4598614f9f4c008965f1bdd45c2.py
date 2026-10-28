import math
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch import nn, Tensor

class SimpleStem(nn.Module):

    def __init__(self, width_in: int, width_out: int, norm_layer: Optional[Callable[..., nn.Module]], activation_layer: Callable[..., nn.Module]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width_in, width_out, kernel_size=3, stride=2, padding=1)
        self.bn = norm_layer(width_out)
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class BasicBlock(nn.Module):

    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width_in, width_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(width_out)
        self.conv2 = nn.Conv2d(width_out, width_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(width_out)
        self.activation = activation_layer(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(width_in, width_out, kernel_size=1, stride=stride, bias=False), norm_layer(width_out)) if width_in != width_out or stride != 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x + identity)
        return x

class AnyStage(nn.Sequential):

    def __init__(self, width_in: int, width_out: int, stride: int, depth: int, block_constructor: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], stage_index: int=0) -> None:
        super().__init__()
        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, stride if i == 0 else 1, norm_layer, activation_layer)
            self.add_module(f'block{stage_index}-{i}', block)
            width_in = width_out

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=0.0001)

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
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
        stem_width: int = 64
        stem_type: Optional[Callable[..., nn.Module]] = SimpleStem
        block_type: Optional[Callable[..., nn.Module]] = BasicBlock
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU
        self.stem = stem_type(in_shape[1], stem_width, norm_layer, activation)
        current_width = stem_width
        blocks = []
        for i, (width_out, stride, depth) in enumerate([(128, 2, 2), (256, 2, 2), (512, 2, 2)]):
            blocks.append((f'block{i + 1}', AnyStage(current_width, width_out, stride, depth, block_type, norm_layer, activation, stage_index=i + 1)))
            current_width = width_out
        self.trunk_output = nn.Sequential(OrderedDict(blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}