from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # Standard choice for CIFAR-10: 3x3 convs, no downsampling via stride in first layer of block if needed, but here we handle stride in first conv
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Projection shortcut if dimensions change
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

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

    _version = 2

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        
        # Input channels from in_shape (usually 3 for RGB)
        in_channels = in_shape[1]
        
        # Architecture: 4 convolution stages, up to 256 channels.
        # Standard CNN for CIFAR-10 often uses 64, 128, 256, 256 or similar.
        # "Residual-style skip connection in deeper feature stages" implies ResNet-like blocks.
        # We will implement a ResNet-like architecture with 4 stages.
        
        # Stage 1: 3x3 conv, BN, ReLU, MaxPool
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: Residual blocks, 64 -> 128
        # "Deeper feature stages" implies residual connections here.
        # Let's use 2 residual blocks in stage 2.
        self.stage2 = nn.Sequential(
            _ResidualBlock(64, 128, stride=1),
            _ResidualBlock(128, 128, stride=1)
        )
        
        # Stage 3: Residual blocks, 128 -> 256
        # Downsample here to reach deeper features
        self.stage3 = nn.Sequential(
            _ResidualBlock(128, 256, stride=2),
            _ResidualBlock(256, 256, stride=1)
        )
        
        # Stage 4: Residual blocks, 256 -> 256
        # "Up to 256 channels"
        self.stage4 = nn.Sequential(
            _ResidualBlock(256, 256, stride=2),
            _ResidualBlock(256, 256, stride=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global Average Pooling
        x = x.mean([2, 3])
        
        x = self.classifier(x)
        return x

    def _load_from_state_dict(
            self,
            state_dict: Dict,
            prefix: str,
            local_metadata: Dict,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: List[str],
    ) -> None:
        version = local_metadata.get("version", None)
        if version not in [1, 2]:
            raise ValueError(f"version should be set to 1 or 2 instead of {version}")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )