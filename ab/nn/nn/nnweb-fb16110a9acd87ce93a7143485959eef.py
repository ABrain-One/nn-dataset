from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_residual = use_residual
        
        if use_residual:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out += self.shortcut(identity)
        out = self.relu(out)
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
        self.num_classes = num_classes
        
        # 4 convolution stages reaching up to 256 channels
        # Stage 1: 3 -> 64
        self.stage1 = nn.Sequential(
            ResidualBlock(3, 64, stride=1, use_residual=False)
        )
        
        # Stage 2: 64 -> 128 (Downsampling)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_residual=True)
        )
        
        # Stage 3: 128 -> 256 (Downsampling)
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_residual=True)
        )
        
        # Stage 4: 256 -> 256 (No downsampling)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 256, stride=1, use_residual=True)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.01)
                init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x


def _squeezenet(
        version: str,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> Net:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = Net(version, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model