from typing import Callable, List, Optional

import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            activation(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_channels),
            activation(),
        )

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


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
        dropout_p: float = prm['dropout']

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU

        # CIFARResidualNet configuration
        # Standard choice for "custom small residual network" on CIFAR-10:
        # Stem: 3x3 conv, no stride (to preserve 32x32)
        # Stage 1: 2 blocks, 32x32, 64 channels
        # Stage 2: 2 blocks, 16x16, 128 channels (stride 2)
        # Stage 3: 2 blocks, 8x8, 256 channels (stride 2)
        
        base_ch = 64
        
        # Stem: 3x3 Conv -> Norm -> Activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(base_ch),
            activation(),
        )

        # Stage 1: 32x32
        self.stage1 = nn.Sequential(
            ResidualBlock(base_ch, base_ch, stride=1, norm_layer=norm_layer, activation=activation, dropout_p=dropout_p),
            ResidualBlock(base_ch, base_ch, stride=1, norm_layer=norm_layer, activation=activation, dropout_p=dropout_p),
        )

        # Stage 2: 16x16
        self.stage2 = nn.Sequential(
            ResidualBlock(base_ch, base_ch * 2, stride=2, norm_layer=norm_layer, activation=activation, dropout_p=dropout_p),
            ResidualBlock(base_ch * 2, base_ch * 2, stride=1, norm_layer=norm_layer, activation=activation, dropout_p=dropout_p),
        )

        # Stage 3: 8x8
        self.stage3 = nn.Sequential(
            ResidualBlock(base_ch * 2, base_ch * 4, stride=2, norm_layer=norm_layer, activation=activation, dropout_p=dropout_p),
            ResidualBlock(base_ch * 4, base_ch * 4, stride=1, norm_layer=norm_layer, activation=activation, dropout_p=dropout_p),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(base_ch * 4, num_classes),
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x