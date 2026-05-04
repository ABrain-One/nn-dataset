# Auto-generated 4-Expert Heterogeneous MoE: AirNet + DPN107 + DarkNet + RegNet
# Four-expert mixture with learned gating, mixup, and LR scheduling

import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from torch import nn, Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _make_divisible, _ovewrite_named_param
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from typing import Any, Callable, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}



# ============================================================================
# EXPERT 1: AirNet
# ============================================================================


class AirInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class AirUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.layers(x)
        return self.relu(x + residual)


class AirNetExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        channels = [64, 128, 256, 512]
        init_block_channels = 64

        self.features = self.build_features(init_block_channels, channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], self.num_classes)

    def build_features(self, init_block_channels, channels):
        layers = [AirInitBlock(self.in_channels, init_block_channels)]
        for i, out_channels in enumerate(channels):
            layers.append(AirUnit(
                in_channels=init_block_channels if i == 0 else channels[i - 1],
                out_channels=out_channels,
                stride=1 if i == 0 else 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()



# ============================================================================
# EXPERT 2: DPN107
# ============================================================================


# Define DPNBlock with Group Convolutions
class DPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=4, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=4, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        return self.relu(out)


# Memory-Optimized DPN107
class DPN107(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, growth_rate):
        super(DPN107, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            *[DPNBlock(growth_rate, growth_rate) for _ in range(num_blocks)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(growth_rate, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DPN107Expert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(DPN107Expert, self).__init__()
        self.device = device
        model_class = DPN107
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]

        self.model = model_class(self.channel_number, self.class_number, num_blocks=3, growth_rate=32)
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.float()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()



# ============================================================================
# EXPERT 3: DarkNet
# ============================================================================


class DarkNetUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pointwise: bool, alpha: float):
        super(DarkNetUnit, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        if pointwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DarkNetExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(DarkNetExpert, self).__init__()
        self.device = device
        channels: list = None
        odd_pointwise: bool = True
        alpha: float = 0.1
        in_channels = in_shape[1]
        image_size = in_shape[2]
        num_classes = out_shape[0]

        if channels is None:
            channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]]

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                pointwise = (len(channels_per_stage) > 1) and not (((j + 1) % 2 == 1) ^ odd_pointwise)
                stage.add_module(f"unit{j + 1}", DarkNetUnit(in_channels, out_channels, pointwise, alpha))
                in_channels = out_channels
            if i != len(channels) - 1:
                stage.add_module(f"pool{i + 1}", nn.MaxPool2d(kernel_size=2, stride=2))
            self.features.add_module(f"stage{i + 1}", stage)

        final_feature_map_size = image_size // (2 ** (len(channels) - 1))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        learning_rate = float(prm.get("lr", 0.01))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.to(self.device)

    def learn(self, train_data: torch.utils.data.DataLoader):
        self.train()
        for inputs, targets in train_data:
            inputs, targets = inputs.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, targets)
            loss.backward()
            self.optimizer.step()



# ============================================================================
# EXPERT 4: RegNet
# ============================================================================



class SimpleStemIN(Conv2dNormActivation):
    def __init__(
            self,
            width_in: int,
            width_out: int,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )


class BottleneckTransform(nn.Sequential):
    def __init__(
            self,
            width_in: int,
            width_out: int,
            stride: int,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
            group_width: int,
            bottleneck_multiplier: float,
            se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = Conv2dNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        layers["b"] = Conv2dNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = Conv2dNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    def __init__(
            self,
            width_in: int,
            width_out: int,
            stride: int,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
            group_width: int = 1,
            bottleneck_multiplier: float = 1.0,
            se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    def __init__(
            self,
            width_in: int,
            width_out: int,
            stride: int,
            depth: int,
            block_constructor: Callable[..., nn.Module],
            norm_layer: Callable[..., nn.Module],
            activation_layer: Callable[..., nn.Module],
            group_width: int,
            bottleneck_multiplier: float,
            se_ratio: Optional[float] = None,
            stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
            self,
            depths: List[int],
            widths: List[int],
            group_widths: List[int],
            bottleneck_multipliers: List[float],
            strides: List[int],
            se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
            cls,
            depth: int,
            w_0: int,
            w_a: float,
            w_m: float,
            group_width: int,
            bottleneck_multiplier: float = 1.0,
            se_ratio: Optional[float] = None
    ) -> "BlockParams":
        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
            stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNetExpert(nn.Module):

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
        block_params: BlockParams = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25)
        num_classes: int = out_shape[0]
        stem_width: int = 32
        stem_type: Optional[Callable[..., nn.Module]] = None
        block_type: Optional[Callable[..., nn.Module]] = None
        norm_layer: Optional[Callable[..., nn.Module]] = None
        activation: Optional[Callable[..., nn.Module]] = None
        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        self.stem = stem_type(
            in_shape[1],
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
                width_out,
                stride,
                depth,
                group_width,
                bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i + 1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def _regnet(
        block_params: BlockParams,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> RegNetExpert:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNetExpert(block_params, norm_layer=norm_layer, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model



# ============================================================================
# HETEROGENEOUS MOE GATE
# ============================================================================
class HeterogeneousGate(nn.Module):
    """Lightweight CNN-based gating network that routes inputs to 4 experts."""
    def __init__(self, input_channels, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.gate_features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x):
        features = self.gate_features(x).flatten(1)
        logits = self.gate(features) / torch.clamp(self.temperature, 0.5, 5.0)
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.1
        return F.softmax(logits, dim=-1), logits


# ============================================================================
# HETEROGENEOUS MOE NET — AirNet + DPN107 + DarkNet + RegNet
# ============================================================================
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        n_experts = 4

        # Defaults for expert-specific hyperparameters
        pass  # no extra defaults needed
        self.experts = nn.ModuleList([
            AirNetExpert(in_shape, out_shape, prm, device),
            DPN107Expert(in_shape, out_shape, prm, device),
            DarkNetExpert(in_shape, out_shape, prm, device),
            RegNetExpert(in_shape, out_shape, prm, device),
        ])
        self.gate = HeterogeneousGate(in_shape[1], n_experts=n_experts)

        self.load_balance_weight = 0.01
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights, _ = self.gate(x)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        return torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=2)

    def _load_balance_loss(self):
        if not hasattr(self, '_last_gw'):
            return torch.tensor(0.0, device=self.device)
        gw = self._last_gw
        usage = gw.sum(dim=0)
        target = gw.sum() / len(self.experts)
        return F.mse_loss(usage, target.expand_as(usage))

    def _mixup_data(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=self.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        expert_params, gate_params = [], list(self.gate.parameters())
        for e in self.experts:
            expert_params.extend(list(e.parameters()))
        self.optimizer = torch.optim.AdamW([
            {'params': expert_params, 'lr': prm.get('lr', 0.001), 'weight_decay': 5e-4},
            {'params': gate_params, 'lr': prm.get('lr', 0.001) * 2, 'weight_decay': 2.5e-4},
        ])
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=5
        )
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self._current_epoch = 0

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            if self.mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = self._mixup_data(inputs, labels)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = lam * self.criteria(outputs, labels_a) + (1 - lam) * self.criteria(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
            self._last_gw = self.gate(inputs)[0].detach()
            loss = loss + self.load_balance_weight * self._load_balance_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        self._current_epoch += 1
        if self._current_epoch <= 5:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()

