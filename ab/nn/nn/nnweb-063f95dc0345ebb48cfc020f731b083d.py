import math
from functools import partial
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """
    2D LayerNorm: normalize over the channel dimension (C) for each spatial location.
    Input shape: (N, C, H, W)
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # LayerNorm expects (..., normalized_shape)
        # We want to normalize over C.
        # Permute to (N, H, W, C) so that C is the last dimension.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt block:
    1. Depthwise Conv 7x7
    2. LayerNorm
    3. MLP (Linear, GELU, Linear)
    4. Stochastic Depth (if prob > 0)
    5. Residual connection
    """
    def __init__(
        self,
        dim: int,
        layer_scale_init: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        # Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        
        # Norm
        self.norm = norm_layer(dim)
        
        # MLP
        hidden_dim = 4 * dim
        self.pwconv1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim, bias=True)
        
        # Layer Scale
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale_init)
        
        # Stochastic Depth
        # We implement a simple Stochastic Depth module
        self.stochastic_depth = nn.Identity()
        if stochastic_depth_prob > 0.0:
            self.stochastic_depth = _StochasticDepth(stochastic_depth_prob)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # 1. Depthwise Conv
        y = self.dwconv(x)
        
        # 2. Norm (handles permute internally)
        y = self.norm(y)
        
        # 3. Permute for Linear (N, H, W, C)
        y = y.permute(0, 2, 3, 1)
        
        # 4. MLP
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)
        
        # 5. Permute back (N, C, H, W)
        y = y.permute(0, 3, 1, 2)
        
        # 6. Layer Scale
        y = self.gamma * y
        
        # 7. Stochastic Depth
        y = self.stochastic_depth(y)
        
        # 8. Residual
        x = x + y
        return x


class _StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        if keep_prob == 0:
            return torch.zeros_like(x)
        mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) > self.drop_prob).float()
        mask = mask / keep_prob
        return x * mask


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
        in_channels: int = in_shape[1]
        
        # Hyperparameters
        stochastic_depth_prob: float = prm['stochastic_depth_prob']
        norm_eps: float = prm['norm_eps']
        norm_std: float = prm['norm_std']
        self.norm_std = norm_std
        
        # Architecture Config
        # CIFAR-10 is 32x32.
        # We downsample 4 times: 32 -> 16 -> 8 -> 4 -> 2
        # Stages:
        # 1. Conv 7x7 s4 -> 8x8, C=96
        # 2. Stage 1: 96, 3 blocks, s1 -> 8x8
        # 3. Downsample: 96->192, s2 -> 4x4
        # 4. Stage 2: 192, 3 blocks, s1 -> 4x4
        # 5. Downsample: 192->384, s2 -> 2x2
        # 6. Stage 3: 384, 3 blocks, s1 -> 2x2
        # 7. Downsample: 384->768, s2 -> 1x1
        # 8. Stage 4: 768, 3 blocks, s1 -> 1x1
        
        # This is a small ConvNeXt.
        # Let's use a slightly deeper config for better accuracy.
        # Small: 96, 192, 384, 768 with 3 blocks each is standard "Small".
        # For CIFAR, maybe fewer blocks but wider? Or standard small.
        # Let's stick to a robust small config.
        
        self.norm_layer = partial(LayerNorm2d, eps=norm_eps)
        
        # Stem
        # Standard ConvNeXt uses 7x7 s4.
        # For 32x32, 7x7 s4 results in 8x8.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=4, padding=3, bias=False),
            self.norm_layer(96)
        )
        
        # Stages
        # Stage 1: 96 ch, 3 blocks
        # Stage 2: 192 ch, 3 blocks
        # Stage 3: 384 ch, 3 blocks
        # Stage 4: 768 ch, 3 blocks
        
        stages_configs = [
            (96, 3, 1),   # dim, num_blocks, stride (stride applied before stage via downsample)
            (192, 3, 2),
            (384, 3, 2),
            (768, 3, 2),
        ]
        
        self.stages = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        
        total_blocks = sum(c[1] for c in stages_configs)
        current_block_idx = 0
        
        prev_dim = 96
        
        for i, (dim, num_blocks, stride) in enumerate(stages_configs):
            # Downsample if needed (except first stage which is handled by stem)
            if i > 0:
                # Downsample from prev_dim to dim with stride
                self.downsamples.append(nn.Sequential(
                    self.norm_layer(prev_dim),
                    nn.Conv2d(prev_dim, dim, kernel_size=1, stride=stride, bias=False)
                ))
            else:
                self.downsamples.append(nn.Identity())
            
            # Create blocks
            stage_blocks = nn.ModuleList([])
            for j in range(num_blocks):
                # Calculate stochastic depth prob
                # Linearly increase from 0 to max_prob
                if total_blocks > 1:
                    sd_prob = stochastic_depth_prob * current_block_idx / (total_blocks - 1)
                else:
                    sd_prob = 0.0
                
                block = ConvNeXtBlock(
                    dim=dim,
                    layer_scale_init=1e-6,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=self.norm_layer
                )
                stage_blocks.append(block)
                current_block_idx += 1
            
            self.stages.append(stage_blocks)
            prev_dim = dim
        
        # Head
        self.head_norm = self.norm_layer(prev_dim)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(prev_dim, num_classes)
        )
        
        # Init
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, std=self.norm_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        
        for i in range(len(self.stages)):
            x = self.downsamples[i](x)
            for block in self.stages[i]:
                x = block(x)
        
        x = self.head_norm(x)
        x = self.head(x)
        
        return x