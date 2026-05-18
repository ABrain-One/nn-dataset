# Auto-generated 4-Expert Heterogeneous MoE: AirNet + BagNet + DPN107 + DenseNet
# Four-expert mixture with learned gating, mixup, and LR scheduling

import numpy as np
from collections import OrderedDict
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from typing import Any, List, Optional, Tuple
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as cp


def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}



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
# EXPERT 2: BagNet
# ============================================================================


class BagNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bottleneck_factor=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = self.conv1x1_block(in_channels, mid_channels)
        self.conv2 = self.conv_block(mid_channels, mid_channels, kernel_size, stride)
        self.conv3 = self.conv1x1_block(mid_channels, out_channels, activation=False)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body = BagNetBottleneck(in_channels, out_channels, kernel_size, stride)

        if self.resize_identity:
            self.identity_conv = self.conv1x1_block(in_channels, out_channels, activation=False)
        self.activ = nn.ReLU(inplace=True)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        if self.resize_identity:
            identity = self.identity_conv(x)

        x = self.body(x)

        if x.size(2) != identity.size(2) or x.size(3) != identity.size(3):
            identity = nn.functional.interpolate(identity, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return self.activ(x + identity)


class BagNetExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        self.channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]]
        self.in_size = image_size
        self.num_classes = class_number

        self.features = nn.Sequential(
            nn.Conv2d(channel_number, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = 64
        for i, stage_channels in enumerate(self.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(stage_channels):
                stride = 2 if (j == 0 and i > 0) else 1
                stage.add_module(f"unit{j + 1}", BagNetUnit(in_channels, out_channels, kernel_size=3, stride=stride))
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
        self.output = nn.Linear(in_channels, self.num_classes)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.output(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'],)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

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
# EXPERT 3: DPN107
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
# EXPERT 4: DenseNet
# ============================================================================



class _DenseLayer(nn.Module):
    def __init__(
            self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input, use_reentrant=False)

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNetExpert(nn.Module):

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
        growth_rate: int = 32
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16)
        num_init_features: int = 64
        bn_size: int = in_shape[0]
        drop_rate: float = prm['dropout']
        memory_efficient: bool = False
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(in_shape[1], num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")

    state_dict = weights.get_state_dict(progress=progress, check_hash=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> DenseNetExpert:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DenseNetExpert(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

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
# HETEROGENEOUS MOE NET — AirNet + BagNet + DPN107 + DenseNet
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
            BagNetExpert(in_shape, out_shape, prm, device),
            DPN107Expert(in_shape, out_shape, prm, device),
            DenseNetExpert(in_shape, out_shape, prm, device),
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

