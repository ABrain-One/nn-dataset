import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.models._utils import _make_divisible
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation


class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int,
            stride: int,
            expand_ratio: float,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        expanded_channels = _make_divisible(in_channels * expand_ratio, 8)
        if expanded_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        squeeze_channels = max(1, in_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        layers.append(
            Conv2dNormActivation(
                expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int,
            stride: int,
            expand_ratio: float,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = _make_divisible(in_channels * expand_ratio, 8)
        if expanded_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=kernel,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            layers.append(
                Conv2dNormActivation(
                    expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


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
        dropout: float = prm['dropout']
        num_classes: int = out_shape[0]
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d

        # EfficientNetV2-S configuration
        # (expand_ratio, kernel_size, stride, in_channels, out_channels, num_layers, block_type)
        # block_type: 0 for FusedMBConv, 1 for MBConv
        settings = [
            (1, 3, 1, 24, 24, 2, 0),
            (4, 3, 2, 24, 48, 4, 0),
            (4, 3, 2, 48, 64, 4, 0),
            (4, 3, 2, 64, 128, 6, 1),
            (6, 3, 1, 128, 160, 9, 1),
            (6, 3, 2, 160, 256, 15, 1),
        ]

        layers: List[nn.Module] = []

        # First conv
        first_in = in_shape[1]
        first_out = settings[0][3]
        layers.append(
            Conv2dNormActivation(
                first_in,
                first_out,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # Calculate total blocks for stochastic depth
        total_blocks = sum(s[5] for s in settings)
        current_block_id = 0

        for expand_ratio, kernel, stride, in_ch, out_ch, num_layers, block_type in settings:
            stage: List[nn.Module] = []
            for i in range(num_layers):
                # Adjust input channels for subsequent blocks in the stage
                if i > 0:
                    cur_in_ch = out_ch
                    cur_stride = 1
                else:
                    cur_in_ch = in_ch
                    cur_stride = stride

                # Stochastic depth probability
                sd_prob = 0.2 * float(current_block_id) / total_blocks

                if block_type == 0:
                    stage.append(
                        FusedMBConv(
                            cur_in_ch,
                            out_ch,
                            kernel,
                            cur_stride,
                            expand_ratio,
                            sd_prob,
                            norm_layer,
                        )
                    )
                else:
                    stage.append(
                        MBConv(
                            cur_in_ch,
                            out_ch,
                            kernel,
                            cur_stride,
                            expand_ratio,
                            sd_prob,
                            norm_layer,
                        )
                    )
                current_block_id += 1
            layers.append(nn.Sequential(*stage))

        last_conv_in = settings[-1][4]
        last_conv_out = 1280

        layers.append(
            Conv2dNormActivation(
                last_conv_in,
                last_conv_out,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_conv_out, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)