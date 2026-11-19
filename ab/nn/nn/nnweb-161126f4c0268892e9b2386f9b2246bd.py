from functools import partial
from typing import Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0, use_se=True, stochastic_depth_prob=0.0, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

        self.stochastic_depth_prob = stochastic_depth_prob

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = self.dropout(out)
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.training and self.stochastic_depth_prob > 0:
            # Stochastic depth: drop entire sample
            keep_prob = 1.0 - self.stochastic_depth_prob
            # Create mask of shape [N, 1, 1, 1]
            mask = torch.bernoulli(torch.full((identity.size(0), 1, 1, 1), keep_prob, device=identity.device))
            out = out * mask / keep_prob if keep_prob > 0 else out

        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out


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
        stochastic_depth_prob: float = prm['stochastic_depth_prob']
        norm_eps: float = prm['norm_eps']
        norm_std: float = prm['norm_std']
        
        # Derive input channels from in_shape
        in_channels = in_shape[1]
        
        # Standard CIFAR-10 ResNet-18 configuration:
        # Stem: 3x3 Conv, no maxpool
        # Stage 1: 2 blocks, 64 channels, 32x32
        # Stage 2: 2 blocks, 128 channels, 16x16
        # Stage 3: 2 blocks, 256 channels, 8x8
        # (Stage 4 is often omitted or merged in small CIFAR variants, but standard ResNet-18 has 4 stages.
        # However, the description says "Stage 1: 32x32, Stage 2: 16x16, Stage 3: 8x8".
        # To match "small residual network" and the specific stage counts in the description,
        # I will implement 3 stages. But wait, ResNet-18 typically has 4 stages of 2 blocks each.
        # The description says "Stage 1... Stage 2... Stage 3...". It implies 3 stages.
        # Let's stick to a 3-stage ResNet-18-like structure (6 blocks total) or standard 4-stage?
        # "Custom small residual network... Stage 1: 32x32, Stage 2: 16x16, Stage 3: 8x8".
        # This implies 3 stages. I will use 2 blocks per stage (total 6 blocks) to keep it "small" and match the depth count often seen in CIFAR variants (like ResNet-14 or 18 adapted).
        # Actually, ResNet-18 has 4 stages. If I use 3 stages, it's ResNet-14 (2 blocks * 3 stages = 6 layers? No, ResNet-14 is 2 blocks * 3 stages? No, ResNet-14 is usually 2 blocks per stage for 3 stages? 
        # Let's look at standard CIFAR ResNet depths:
        # ResNet-14: 2 blocks per stage, 3 stages? No, ResNet-14 is 2*3 + 2 = 8 layers? 
        # Standard ResNet depth n = 6k + 2.
        # ResNet-14: k=2. 2 blocks per stage. 3 stages? 2*3=6 blocks + 1 conv = 7 layers? 
        # ResNet-18: k=3. 3 blocks per stage? No, ResNet-18 is 2 blocks per stage, 4 stages. 2*4=8 blocks + 1 conv = 9 layers? 
        # Wait, ResNet-18 on ImageNet is 18 layers. On CIFAR, it's often adapted.
        # The prompt says "CIFARResidualNet... reported 93.78%". This is a high accuracy, suggesting a decent depth.
        # I will implement a 3-stage network with 2 blocks per stage (Total 6 basic blocks). This is a common "small" variant.
        # Widths: 64, 128, 256.
        
        base_width = 64
        depths = [2, 2, 2] # 2 blocks per stage
        
        norm_layer = partial(LayerNorm2d, eps=norm_eps)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(base_width, base_width, depths[0], stride=1, norm_layer=norm_layer, dropout_prob=0.0, use_se=True, sd_prob=0.0)
        self.layer2 = self._make_layer(base_width, base_width * 2, depths[1], stride=2, norm_layer=norm_layer, dropout_prob=0.0, use_se=True, sd_prob=stochastic_depth_prob / 2)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, depths[2], stride=2, norm_layer=norm_layer, dropout_prob=0.0, use_se=True, sd_prob=stochastic_depth_prob)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=norm_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, out_channels, depth, stride, norm_layer, dropout_prob, use_se, sd_prob):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout_prob, use_se, sd_prob, norm_layer))
        for _ in range(1, depth):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout_prob, use_se, sd_prob, norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)