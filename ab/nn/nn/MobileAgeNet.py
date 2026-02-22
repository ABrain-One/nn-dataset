# MobileAgeNet: MobileNetV3-based lightweight CNN for age estimation (~1.5M params, ONNX/TFLite ready)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


# Hard-Swish: mobile-efficient activation
class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0) / 6.0


# Hard-Sigmoid: used in SE blocks
class HardSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0) / 6.0


# Lightweight channel attention via squeeze-and-excitation
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.activation = HardSigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = self.activation(self.fc2(scale))
        return x * scale


# MobileNetV3-style inverted residual with optional SE and depthwise separable conv
class InvertedResidual(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_factor: float = 4,
        use_se: bool = True
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_channels = int(in_channels * expansion_factor)
        padding = (kernel_size - 1) // 2
        
        layers = []
        if expansion_factor != 1:
            layers.extend([nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                           nn.BatchNorm2d(hidden_channels), HardSwish()])
        layers.extend([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=hidden_channels, bias=False),
                       nn.BatchNorm2d(hidden_channels), HardSwish()])
        if use_se:
            layers.append(SqueezeExcitation(hidden_channels))
        layers.extend([nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels)])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# Main network: stem -> 4 inverted residual stages -> 1x1 conv -> regression head
class Net(nn.Module):
    def train_setup(self, prm: Dict[str, Any]) -> None:
        self.to(self.device)
        self.criteria = (nn.L1Loss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=1e-4
        )
    
    def learn(self, train_data: Any) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()
    
    def __init__(self, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...],
                 prm: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout = prm.get('dropout', 0.2)
        in_channels = in_shape[1] if len(in_shape) >= 2 else 3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        self.stage1 = nn.Sequential(  # 16 -> 24, stride 2
            InvertedResidual(16, 24, kernel_size=3, stride=2, expansion_factor=4, use_se=True),
            InvertedResidual(24, 24, kernel_size=3, stride=1, expansion_factor=3, use_se=True),
        )
        self.stage2 = nn.Sequential(  # 24 -> 40, stride 2
            InvertedResidual(24, 40, kernel_size=5, stride=2, expansion_factor=3, use_se=True),
            InvertedResidual(40, 40, kernel_size=5, stride=1, expansion_factor=3, use_se=True),
            InvertedResidual(40, 40, kernel_size=5, stride=1, expansion_factor=3, use_se=True),
        )
        self.stage3 = nn.Sequential(  # 40 -> 112, stride 2
            InvertedResidual(40, 80, kernel_size=3, stride=2, expansion_factor=6, use_se=False),
            InvertedResidual(80, 80, kernel_size=3, stride=1, expansion_factor=2.5, use_se=False),
            InvertedResidual(80, 80, kernel_size=3, stride=1, expansion_factor=2.3, use_se=False),
            InvertedResidual(80, 112, kernel_size=3, stride=1, expansion_factor=6, use_se=True),
        )
        self.stage4 = nn.Sequential(  # 112 -> 160, stride 2
            InvertedResidual(112, 160, kernel_size=5, stride=2, expansion_factor=6, use_se=True),
            InvertedResidual(160, 160, kernel_size=5, stride=1, expansion_factor=6, use_se=True),
        )
        self.final_conv = nn.Sequential(  # expand to 960 channels before pooling
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish()
        )
        
        # Regression head: pool -> FC(256) -> dropout -> FC(1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(960, 256), HardSwish(), nn.Dropout(p=dropout),
            nn.Linear(256, out_shape[0])
        )
        self._initialize_weights()

    # Kaiming init for convs, normal init for linear layers
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_conv(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
