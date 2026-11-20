from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _make_divisible, _ovewrite_named_param
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer


class InvertedResidualConfig:
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(
            self,
            cnf: InvertedResidualConfig,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'norm_eps', 'norm_momentum'}


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
        
        # Hyperparameters
        dropout_p = prm['dropout']
        norm_eps = prm['norm_eps']
        norm_momentum = prm['norm_momentum']
        
        # Architecture parameters based on description:
        # "4 convolution stages reaching up to 256 channels"
        # "Batch normalization after convolution blocks"
        # "Residual-style skip connection in deeper feature stages"
        
        # We interpret "4 convolution stages" as 4 blocks of convolutions.
        # To reach 256 channels, we can use widths like [64, 128, 256, 256] or similar.
        # Standard choice for CIFAR-10 CNNs often uses smaller kernels (3x3) and stride 2 for downsampling.
        # Let's design a simple CNN with 4 stages.
        # Stage 1: 3 -> 64, 3x3, stride 1, pad 1
        # Stage 2: 64 -> 128, 3x3, stride 2, pad 1
        # Stage 3: 128 -> 256, 3x3, stride 2, pad 1
        # Stage 4: 256 -> 256, 3x3, stride 1, pad 1 (Deeper feature stage with residual)
        
        # The description says "Residual-style skip connection in deeper feature stages".
        # We will implement a simple residual block for the later stages.
        
        norm_layer = partial(nn.BatchNorm2d, eps=norm_eps, momentum=norm_momentum)
        
        # Helper to create a conv block: Conv -> BN -> ReLU
        def conv_block(in_ch, out_ch, stride=1, padding=1, use_residual=False, in_ch_res=None):
            layers = []
            if use_residual:
                # If channels or spatial dims change, we need a projection shortcut
                identity = nn.Identity()
                if in_ch != out_ch or stride != 1:
                    identity = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        norm_layer(out_ch)
                    )
                self.identity = identity
                
                conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=False)
                bn1 = norm_layer(out_ch)
                relu1 = nn.ReLU(inplace=True)
                conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=False)
                bn2 = norm_layer(out_ch)
                
                layers.append(conv1)
                layers.append(bn1)
                layers.append(relu1)
                layers.append(conv2)
                layers.append(bn2)
                
                # Return a module that handles the residual add
                return nn.Sequential(*layers), identity
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=False))
                layers.append(norm_layer(out_ch))
                layers.append(nn.ReLU(inplace=True))
                return nn.Sequential(*layers), None

        # Build the 4 stages
        # Input channels from in_shape[1]
        in_ch = in_shape[1]
        
        # Stage 1: 3x3, stride 1
        stage1, _ = conv_block(in_ch, 64, stride=1, padding=1, use_residual=False)
        
        # Stage 2: 3x3, stride 2
        stage2, _ = conv_block(64, 128, stride=2, padding=1, use_residual=False)
        
        # Stage 3: 3x3, stride 2
        stage3, _ = conv_block(128, 256, stride=2, padding=1, use_residual=False)
        
        # Stage 4: "Deeper feature stage" with residual
        # We can have multiple residual blocks here. Let's do 2 blocks of 256 channels.
        # Block 4a
        block4a, id4a = conv_block(256, 256, stride=1, padding=1, use_residual=True)
        # Block 4b
        block4b, id4b = conv_block(256, 256, stride=1, padding=1, use_residual=True)
        
        self.features = nn.Sequential(
            stage1,
            stage2,
            stage3,
            block4a,
            block4b
        )
        
        # We need to handle the residual connections properly.
        # The simple sequential above doesn't allow adding identity.
        # Let's re-structure using a custom module or just explicit layers.
        
        # Re-implementing features to support residuals correctly.
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        
        # For stage 4, we need to apply the blocks with residuals.
        # We will create a small module for the residual blocks.
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, norm_layer=None):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = norm_layer(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = norm_layer(out_ch)
                
                self.shortcut = nn.Identity()
                if in_ch != out_ch or stride != 1:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        norm_layer(out_ch)
                    )
                
            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = self.relu(out)
                return out

        # We will use ResidualBlock for the "deeper feature stages" (Stage 4).
        # Let's assume Stage 4 consists of 2 ResidualBlocks.
        self.res_block1 = ResidualBlock(256, 256, stride=1, norm_layer=norm_layer)
        self.res_block2 = ResidualBlock(256, 256, stride=1, norm_layer=norm_layer)
        
        # Classifier
        # Input to classifier is the output of the last feature map.
        # We use AdaptiveAvgPool2d to handle variable sizes if needed, though CIFAR is fixed.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_shape[0])
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Deeper feature stages with residuals
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _mobilenet_v3_conf(
        arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> Net:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = Net(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model