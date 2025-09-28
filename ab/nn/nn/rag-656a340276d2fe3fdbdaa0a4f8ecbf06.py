# Auto-generated single-file for _InvertedResidual
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
from torch import Tensor
import torch

# ---- original imports from contributing modules ----

# ---- _InvertedResidual (target) ----
class _InvertedResidual(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, stride: int, expansion_factor: int, bn_momentum: float = 0.1
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        if kernel_size not in [3, 5]:
            raise ValueError(f"kernel_size should be 3 or 5 instead of {kernel_size}")
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)

def supported_hyperparameters():
    return ['lr', 'momentum']

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        
        self.inverted_residual = _InvertedResidual(
            in_ch=64,
            out_ch=128,
            kernel_size=3,
            stride=1,
            expansion_factor=6,
            bn_momentum=0.1
        )
        self.classifier = nn.Linear(128 * 4 * 4, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        
        x = self.inverted_residual(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        import torch
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        import torch
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
