
from functools import partial
from typing import Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_layer=None, stochastic_depth_prob: float = 0.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(channels)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "batch")

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.stochastic_depth(out)
        out = out + identity
        return F.relu(out)


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.9)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = norm_layer(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    def __init__(
            self,
            input_channels: int,
            out_channels: Optional[int],
            num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


def supported_hyperparameters():
    return {'lr', 'momentum', 'stochastic_depth_prob', 'norm_eps', 'norm_std'}


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
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        norm_eps: float = prm['norm_eps']
        norm_std: float = prm['norm_std']
        stochastic_depth_prob: float = prm['stochastic_depth_prob']

        norm_layer = partial(nn.BatchNorm2d, eps=norm_eps, momentum=0.9)

        # Stochastic depth schedule: linearly ramp from 0 to stochastic_depth_prob
        # across the residual blocks (stages 1-4: 2+2+2+2 = 8 blocks)
        num_blocks = 8
        sd_probs = [stochastic_depth_prob * i / num_blocks for i in range(num_blocks)]

        # Stem: 3x3 conv stride 2 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 -> 64, 2 blocks, 32x32
        stage1 = nn.Sequential(
            ResidualBlock(64, norm_layer, sd_probs[0]),
            ResidualBlock(64, norm_layer, sd_probs[1]),
        )

        # Stage 2: 64 -> 128, 2 blocks, 16x16
        stage2 = nn.Sequential(
            DownsampleBlock(64, 128, norm_layer),
            ResidualBlock(128, norm_layer, sd_probs[2]),
            ResidualBlock(128, norm_layer, sd_probs[3]),
        )

        # Stage 3: 128 -> 256, 2 blocks, 8x8
        stage3 = nn.Sequential(
            DownsampleBlock(128, 256, norm_layer),
            ResidualBlock(256, norm_layer, sd_probs[4]),
            ResidualBlock(256, norm_layer, sd_probs[5]),
        )

        # Stage 4: 256 -> 512, 2 blocks, 4x4
        stage4 = nn.Sequential(
            DownsampleBlock(256, 512, norm_layer),
            ResidualBlock(512, norm_layer, sd_probs[6]),
            ResidualBlock(512, norm_layer, sd_probs[7]),
        )

        self.features = nn.Sequential(stage1, stage2, stage3, stage4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=norm_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
