# Auto-generated single-file for FFN
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Union, Callable
from typing import Type
from typing import Callable
from typing import Union

# ---- timm.models.shvit.Conv2dNorm ----
class Conv2dNorm(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            bn_weight_init: int = 1,
            **kwargs,
    ):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    def fuse(self) -> nn.Conv2d:
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            in_channels=w.size(1) * self.c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
            dtype=c.weight.dtype,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# ---- timm.layers.typing.LayerType ----
LayerType = Union[str, Callable, Type[torch.nn.Module]]

# ---- FFN (target) ----
class FFN(nn.Module):
    def __init__(self, dim: int, embed_dim: int, act_layer: LayerType = nn.ReLU):
        super().__init__()
        self.pw1 = Conv2dNorm(dim, embed_dim)
        self.act = act_layer()
        self.pw2 = Conv2dNorm(embed_dim, dim, bn_weight_init=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x

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
        self.ffn = FFN(dim=32, embed_dim=64, act_layer=nn.ReLU)
        self.classifier = nn.Linear(32, self.num_classes)

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
        x = self.ffn(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
