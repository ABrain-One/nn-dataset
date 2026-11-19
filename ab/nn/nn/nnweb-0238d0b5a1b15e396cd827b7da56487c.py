import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'dropout_aux'}


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
        num_classes = out_shape[0]
        in_channels = in_shape[1]
        
        # Hyperparameters
        dropout = prm['dropout']
        dropout_aux = prm['dropout_aux']
        
        # Standard choices for CIFAR-10 (32x32 input)
        # Input: 32x32
        # Stem: 3x3 Conv, Norm, Act -> 32x32
        # Stage 1: ResBlocks at 32x32
        # Stage 2: ResBlocks at 16x16 (downsample)
        # Stage 3: ResBlocks at 8x8 (downsample)
        
        # Widths: Standard small ResNet choices (e.g., width 32 or 64). 
        # Let's use width 32 for a "small" network, or 64 for robustness. 
        # The description says "custom small residual network". 
        # Let's pick width 32.
        base_width = 32
        
        # Depth: "Residual blocks" implies some number. 
        # Standard ResNet-18 has [2, 2, 2, 2]. 
        # Here we have 3 stages. Let's use [2, 2, 2] blocks per stage.
        depths = [2, 2, 2]
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(base_width, base_width, depths[0], downsample=False, dropout=dropout, dropout_aux=dropout_aux)
        self.stage2 = self._make_stage(base_width, base_width * 2, depths[1], downsample=True, dropout=dropout, dropout_aux=dropout_aux)
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, depths[2], downsample=True, dropout=dropout, dropout_aux=dropout_aux)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use dropout_aux in the classifier head to satisfy the hyperparameter usage requirement
        # typically dropout is used in blocks, dropout_aux in head/aux
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_aux),
            nn.Linear(base_width * 4, num_classes)
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, downsample: bool, dropout: float, dropout_aux: float) -> nn.Sequential:
        layers = []
        # First block handles downsampling if needed
        layers.append(ResidualBlock(in_channels, out_channels, stride=2 if downsample else 1, dropout=dropout, dropout_aux=dropout_aux))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout=dropout, dropout_aux=dropout_aux))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0, dropout_aux: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Optional Dropout2d as per description
        self.dropout = nn.Dropout2d(p=dropout)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        # Note: dropout_aux is passed but typically used in head. 
        # To strictly "use" it in the block if required by strict interpretation, 
        # we could add it here, but standard practice puts it in head. 
        # The prompt says "each of those names... must still be read from prm and actually used in the new model".
        # I used dropout in the block. I used dropout_aux in the head. 
        # This satisfies the condition that both are read and used somewhere in the model.
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        out += identity
        out = self.relu(out)
        return out