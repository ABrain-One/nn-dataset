# Auto-generated single-file for DownsampleConv
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional, Callable
from typing import Callable
from typing import Optional

# ---- original imports from contributing modules ----
from typing import Callable, Optional

# ---- DownsampleConv (target) ----
class DownsampleConv(nn.Module):
    """1x1 convolution downsampling module."""

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            preact: bool = True,
            conv_layer: Optional[Callable] = None,
            norm_layer: Optional[Callable] = None,
    ):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Downsampled tensor.
        """
        return self.norm(self.conv(x))

def supported_hyperparameters():
    return {'lr', 'momentum'}

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        self.downsample_conv = DownsampleConv(
            in_chs=32,
            out_chs=32,
            stride=1,
            dilation=1,
            first_dilation=None,
            preact=True,
            conv_layer=nn.Conv2d,
            norm_layer=nn.BatchNorm2d
        )

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
