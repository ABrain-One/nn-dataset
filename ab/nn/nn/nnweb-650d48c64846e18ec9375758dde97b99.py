import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div_(keep_prob) * random_tensor
        return x


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(out_ch)
        self.act = activation_layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DepthwiseConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias,
        )
        self.norm = norm_layer(out_ch)
        self.act = activation_layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """A standard 3x3 convolution block with residual connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        drop_path_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.use_residual = in_ch == out_ch and stride == 1

        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                norm_layer(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.block = nn.Sequential(
            ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm_layer=norm_layer, activation_layer=activation_layer),
            ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm_layer=norm_layer, activation_layer=activation_layer),
        )
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        out = self.block(x)
        out = self.drop_path(out)
        return shortcut + out


class SE(nn.Module):
    """Squeeze and Excitation block."""

    def __init__(self, in_ch: int, reduction_ratio: float = 0.25):
        super().__init__()
        mid_ch = max(8, int(in_ch * reduction_ratio))
        self.fc1 = nn.Linear(in_ch, mid_ch)
        self.fc2 = nn.Linear(mid_ch, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = x.mean(dim=(2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class MBConvBlock(nn.Module):
    """MobileNetV3 style block with SE and drop path."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        expansion_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_path_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.use_residual = in_ch == out_ch and stride == 1
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                norm_layer(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        mid_ch = in_ch * expansion_ratio
        self.block = nn.Sequential(
            ConvNormAct(in_ch, mid_ch, 1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer),
            DepthwiseConvNormAct(mid_ch, mid_ch, kernel_size, stride=stride, padding=kernel_size // 2, norm_layer=norm_layer, activation_layer=activation_layer),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            norm_layer(out_ch),
        )
        self.se = SE(out_ch, se_ratio)
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        out = self.block(x)
        out = self.se(out)
        out = self.drop_path(out)
        return shortcut + out


class RegNetBlock(nn.Module):
    """RegNet style block with optional SE and drop path."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
        drop_path_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.use_residual = in_ch == out_ch and stride == 1
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                norm_layer(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.block = nn.Sequential(
            ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, norm_layer=norm_layer, activation_layer=activation_layer),
            ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2, norm_layer=norm_layer, activation_layer=activation_layer),
        )
        self.se = SE(out_ch, se_ratio) if se_ratio > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        out = self.block(x)
        out = self.se(out)
        out = self.drop_path(out)
        return shortcut + out


class EfficientNet(nn.Module):
    """
    A simplified EfficientNet-style architecture with MBConv blocks, SE, and drop path.
    Designed for CIFAR-10 (32x32x3).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        drop_path_prob: float = 0.1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ):
        super().__init__()

        # Stem
        self.stem = ConvNormAct(in_channels, 32, 3, stride=1, padding=1, norm_layer=norm_layer, activation_layer=activation_layer)

        # Block configuration: (out_channels, num_blocks, kernel_size, stride, expansion_ratio, se_ratio)
        # This is a lightweight configuration suitable for CIFAR-10
        block_config = [
            # (out_ch, num_blocks, kernel_size, stride, expansion_ratio, se_ratio)
            (16, 1, 3, 1, 1, 0.0),
            (32, 2, 3, 2, 6, 0.25),
            (64, 2, 3, 2, 6, 0.25),
            (128, 3, 3, 2, 6, 0.25),
            (256, 4, 3, 1, 6, 0.25),
        ]

        self.blocks = nn.ModuleList()
        current_ch = 32
        total_blocks = sum(cfg[1] for cfg in block_config)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_prob, total_blocks)]

        block_idx = 0
        for out_ch, num_blocks, kernel_size, stride, expansion_ratio, se_ratio in block_config:
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        in_ch=current_ch,
                        out_ch=out_ch,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        se_ratio=se_ratio,
                        drop_path_prob=drop_path_rates[block_idx],
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                    )
                )
                current_ch = out_ch
                block_idx += 1

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}


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
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        dropout = prm['dropout']
        stochastic_depth_prob = prm['stochastic_depth_prob']
        attention_dropout = prm['attention_dropout']  # Not used in this architecture but must be read

        norm_layer = nn.BatchNorm2d
        activation_layer = nn.SiLU

        self.model = EfficientNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            drop_path_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)