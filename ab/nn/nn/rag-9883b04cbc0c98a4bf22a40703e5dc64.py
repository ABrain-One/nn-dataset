# Auto-generated single-file for InceptionDWConv2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import List, Union
import collections
from itertools import repeat
from typing import List
from typing import Union
from collections import *

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.layers.padding.get_padding ----
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> Union[int, List[int]]:
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

# ---- InceptionDWConv2d (target) ----
class InceptionDWConv2d(nn.Module):
    """ Inception depthwise convolution
    """

    def __init__(
            self,
            in_chs,
            square_kernel_size=3,
            band_kernel_size=11,
            branch_ratio=0.125,
            dilation=1,
    ):
        super().__init__()

        gc = int(in_chs * branch_ratio)  # channel numbers of a convolution branch
        square_padding = get_padding(square_kernel_size, dilation=dilation)
        band_padding = get_padding(band_kernel_size, dilation=dilation)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size,
            padding=square_padding, dilation=dilation, groups=gc)
        self.dwconv_w = nn.Conv2d(
            gc, gc, (1, band_kernel_size),
            padding=(0, band_padding), dilation=(1, dilation), groups=gc)
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1),
            padding=(band_padding, 0), dilation=(dilation, 1), groups=gc)
        self.split_indexes = (in_chs - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((
            x_id,
            self.dwconv_hw(x_hw),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
            ), dim=1,
        )

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
        self.inception_dwconv2d = InceptionDWConv2d(in_chs=32, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125, dilation=1)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.inception_dwconv2d(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
