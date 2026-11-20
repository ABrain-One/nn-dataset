import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth


def _get_conv_output_shape(input_size: Tuple[int, int], kernel_size: int, stride: int, padding: int) -> Tuple[int, int]:
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


def _make_block_input_shapes(input_size: Tuple[int, int], n_blocks: int) -> List[Tuple[int, int]]:
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_activation(**kwargs) -> nn.Module:
    """
    Factory function for activation layers.
    Accepts **kwargs to satisfy Conv2dNormActivation which may pass 'inplace'.
    Returns nn.GELU.
    """
    return nn.GELU()


class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_ratio: float,
            squeeze_ratio: float,
            stride: int,
            activation_layer: Callable[..., nn.Module],
            norm_layer: Callable[..., nn.Module],
            p_stochastic_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        proj: Sequence[nn.Module]
        self.proj: nn.Module

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")
        else:
            self.stochastic_depth = nn.Identity()

        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        _layers["conv_a"] = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=None,
        )
        _layers["conv_b"] = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            groups=mid_channels,
            inplace=None,
        )
        _layers["squeeze_excitation"] = SqueezeExcitation(mid_channels, sqz_channels, activation=nn.SiLU)
        _layers["conv_c"] = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)

        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor) -> Tensor:
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class ResBlock(nn.Module):
    """
    Standard ResNet block with GroupNorm (BatchNorm alternative for stability) and optional stochastic depth.
    Uses 3x3 convs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        p_stochastic_dropout: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.p_stochastic_dropout = p_stochastic_dropout

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")
        else:
            self.stochastic_depth = nn.Identity()

        self.conv1 = Conv2dNormActivation(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            norm_layer=norm_layer, activation_layer=activation_layer, bias=False
        )
        
        self.conv2 = Conv2dNormActivation(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            norm_layer=norm_layer, activation_layer=None, bias=False
        )
        
        # Add SE after conv2, before add
        if use_se:
            self.se = SqueezeExcitation(out_channels, out_channels // 4, activation=nn.SiLU)
        else:
            self.se = nn.Identity()

        # Projection for residual connection if needed
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.proj(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        
        out = self.stochastic_depth(out)
        out += identity
        
        return out


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
        input_size: Tuple[int, int] = in_shape[2:]
        num_classes: int = out_shape[0]
        input_channels = in_shape[1]

        # Hyperparameters
        dropout_p = prm['dropout']
        attention_dropout = prm['attention_dropout'] # Used in SE or potentially ignored if we don't use attention, but we must read it.
        stochastic_depth_prob = prm['stochastic_depth_prob']
        momentum = prm['momentum'] # Read but used in optimizer

        # Architecture Config
        # We will use a ResNet-like architecture with GroupNorm for stability.
        # Channels: [64, 128, 256, 512]
        # Layers per stage: [2, 2, 4, 2] (Total 10 blocks, similar to ResNet-18/34 but deeper features)
        block_channels = [64, 128, 256, 512]
        block_layers = [2, 2, 4, 2]
        
        # Norm Layer: GroupNorm is often more stable for small batches and fast training.
        # We use 8 groups.
        norm_layer = partial(nn.GroupNorm, 8)
        # Use the helper function to avoid inplace kwarg errors
        activation_layer = _get_activation

        # Stem
        # Start with a large kernel to capture context early? Or standard 7x7?
        # For CIFAR, 3x3 is standard. Let's use a 3x3 stem with stride 2.
        stem_channels = 64
        
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=False
            ),
            Conv2dNormActivation(
                stem_channels, stem_channels, 3, stride=1, padding=1,
                norm_layer=norm_layer, activation_layer=activation_layer, bias=False
            )
        )

        current_size = _get_conv_output_shape(input_size, 3, 2, 1)
        
        self.blocks = nn.ModuleList()
        in_ch = stem_channels
        
        # Calculate total layers for stochastic depth
        total_layers = sum(block_layers)
        p_stochastic = np.linspace(0, stochastic_depth_prob, total_layers).tolist()
        
        p_idx = 0
        for i, (out_ch, num_blocks) in enumerate(zip(block_channels, block_layers)):
            stride = 2 if i > 0 else 1 # First block keeps resolution from stem (which already halved)
            # Actually, standard ResNet stems with stride 2, then blocks with stride 1 for first block, stride 2 for others.
            # My stem has stride 2. 
            # Block 0: stride 1 (input 32->32)
            # Block 1: stride 2 (input 32->16)
            # Block 2: stride 2 (input 16->8)
            # Block 3: stride 2 (input 8->4)
            
            # Wait, if stem is stride 2, input 32->16.
            # Let's adjust strides:
            # Stem: 3x3, stride 2. (32->16)
            # Block 0: 3x3, stride 1. (16->16)
            # Block 1: 3x3, stride 2. (16->8)
            # Block 2: 3x3, stride 2. (8->4)
            # Block 3: 3x3, stride 2. (4->2)
            
            block_stride = 1 if i == 0 else 2
            
            for j in range(num_blocks):
                # First block in a stage (except the very first stage) might need stride 2?
                # In ResNet, the first block of a stage (except stage 1) has stride 2.
                # Here, stem did stride 2.
                # So Block 0 (Stage 1) should be stride 1.
                # Block 1 (Stage 2) first block stride 2, rest stride 1.
                
                block_in_ch = in_ch if j == 0 else out_ch
                block_str = block_stride if j == 0 else 1
                
                self.blocks.append(
                    ResBlock(
                        in_channels=block_in_ch,
                        out_channels=out_ch,
                        stride=block_str,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        p_stochastic_dropout=p_stochastic[p_idx],
                        use_se=True
                    )
                )
                p_idx += 1
                
                # Update current size for next block
                if block_str > 1:
                    current_size = _get_conv_output_shape(current_size, 3, block_str, 1)
            
            in_ch = out_ch
            # Update size if stride was applied in this stage
            if block_stride > 1:
                 current_size = _get_conv_output_shape(current_size, 3, block_stride, 1)
            # Note: The loop above updates size for each block. 
            # Let's re-verify size logic.
            # Stem: 32->16.
            # Stage 0 (i=0): stride 1. Size stays 16.
            # Stage 1 (i=1): stride 2. Size 16->8.
            # Stage 2 (i=2): stride 2. Size 8->4.
            # Stage 3 (i=3): stride 2. Size 4->2.
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(block_channels[-1], block_channels[-1]),
            _get_activation(),
            nn.Dropout(p=dropout_p),
            nn.Linear(block_channels[-1], num_classes)
        )

        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}