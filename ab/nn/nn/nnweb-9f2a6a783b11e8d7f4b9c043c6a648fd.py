import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        use_se: bool = False,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.se = None
        if use_se:
            self.se = SqueezeExcitation(out_channels, out_channels // 4)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row") if stochastic_depth_prob > 0 else nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            out = self.se(out)
            
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.stochastic_depth(out)
        out = out + identity

        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, rd: int = 16):
        super().__init__()
        rd_channels = max(1, in_channels // rd)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, rd_channels),
            nn.ReLU(),
            nn.Linear(rd_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str = "row"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.mode == "row":
            mask = torch.empty((x.size(0), 1), dtype=x.dtype, device=x.device).bernoulli(1.0 - self.p)
        else:
            mask = torch.empty((x.size(0), x.size(1), x.size(2), x.size(3)), dtype=x.dtype, device=x.device).bernoulli(1.0 - self.p)
        return x * mask


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
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.dropout = prm['dropout']
        
        # Standard choices for "small residual network" on CIFAR-10
        # Depth: 2 blocks per stage is standard for "small" (e.g., ResNet-14 or similar)
        # Width: 64 is a common base width for CIFAR
        base_width = 64
        blocks_per_stage = 2
        
        norm_layer = nn.BatchNorm2d
        
        # Stem: 3x3 Conv -> Norm -> Activation
        # For CIFAR (32x32), we usually do NOT stride the stem to keep spatial size 32x32 for Stage 1
        # However, the description says:
        # Stem: 3x3 Conv2d, norm, activation.
        # Stage 1: residual blocks at 32x32.
        # Stage 2: residual blocks at 16x16.
        # Stage 3: residual blocks at 8x8.
        # This implies the stem does NOT downsample (stride 1), or Stage 1 starts at 32x32.
        # If stem is stride 1, output is 32x32.
        # Stage 1 (32x32): Blocks with stride 1.
        # Stage 2 (16x16): First block has stride 2.
        # Stage 3 (8x8): First block has stride 2.
        
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(base_width),
            nn.ReLU()
        )
        
        self.stages = nn.Sequential()
        current_channels = base_width
        
        # Stage 1: 32x32
        stage1_blocks = []
        for i in range(blocks_per_stage):
            stage1_blocks.append(
                BasicBlock(
                    current_channels,
                    current_channels,
                    stride=1,
                    norm_layer=norm_layer,
                    dropout=self.dropout,
                    stochastic_depth_prob=0.0, # Optional, keeping 0 for simplicity unless specified
                    use_se=False # Optional, keeping False for simplicity unless specified
                )
            )
        self.stages.add_module("stage1", nn.Sequential(*stage1_blocks))
        
        # Stage 2: 16x16
        next_channels = base_width * 2
        stage2_blocks = []
        for i in range(blocks_per_stage):
            stride = 2 if i == 0 else 1
            in_ch = current_channels if i == 0 else next_channels
            stage2_blocks.append(
                BasicBlock(
                    in_ch,
                    next_channels,
                    stride=stride,
                    norm_layer=norm_layer,
                    dropout=self.dropout,
                    stochastic_depth_prob=0.0,
                    use_se=False
                )
            )
        self.stages.add_module("stage2", nn.Sequential(*stage2_blocks))
        current_channels = next_channels
        
        # Stage 3: 8x8
        next_channels = base_width * 4
        stage3_blocks = []
        for i in range(blocks_per_stage):
            stride = 2 if i == 0 else 1
            in_ch = current_channels if i == 0 else next_channels
            stage3_blocks.append(
                BasicBlock(
                    in_ch,
                    next_channels,
                    stride=stride,
                    norm_layer=norm_layer,
                    dropout=self.dropout,
                    stochastic_depth_prob=0.0,
                    use_se=False
                )
            )
        self.stages.add_module("stage3", nn.Sequential(*stage3_blocks))
        current_channels = next_channels
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, self.num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x