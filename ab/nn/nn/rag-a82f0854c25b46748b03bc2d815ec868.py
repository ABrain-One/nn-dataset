# Auto-generated single-file for SelectAdaptivePool2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from enum import Enum
from typing import Union
from typing import Tuple

# ---- timm.layers.format.Format ----
class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

# ---- timm.layers.adaptive_avgmax_pool.adaptive_pool_feat_mult ----
def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1

# ---- timm.layers.adaptive_avgmax_pool._int_tuple_2_t ----
_int_tuple_2_t = Union[int, Tuple[int, int]]

# ---- timm.layers.adaptive_avgmax_pool.adaptive_avgmax_pool2d ----
def adaptive_avgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)

# ---- timm.layers.adaptive_avgmax_pool.AdaptiveAvgMaxPool2d ----
class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)

# ---- timm.layers.adaptive_avgmax_pool.adaptive_catavgmax_pool2d ----
def adaptive_catavgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)

# ---- timm.layers.adaptive_avgmax_pool.AdaptiveCatAvgMaxPool2d ----
class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)

# ---- timm.layers.format.FormatT ----
FormatT = Union[str, Format]

# ---- timm.layers.format.get_spatial_dim ----
def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim

# ---- timm.layers.adaptive_avgmax_pool.FastAdaptiveAvgMaxPool ----
class FastAdaptiveAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim, keepdim=not self.flatten)
        x_max = x.amax(self.dim, keepdim=not self.flatten)
        return 0.5 * x_avg + 0.5 * x_max

# ---- timm.layers.adaptive_avgmax_pool.FastAdaptiveAvgPool ----
class FastAdaptiveAvgPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: F = 'NCHW'):
        super(FastAdaptiveAvgPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.mean(self.dim, keepdim=not self.flatten)

# ---- timm.layers.adaptive_avgmax_pool.FastAdaptiveMaxPool ----
class FastAdaptiveMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.amax(self.dim, keepdim=not self.flatten)

# ---- timm.layers.format.get_channel_dim ----
def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim

# ---- timm.layers.adaptive_avgmax_pool.FastAdaptiveCatAvgMaxPool ----
class FastAdaptiveCatAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveCatAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)
        if flatten:
            self.dim_cat = 1
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim_reduce, keepdim=not self.flatten)
        x_max = x.amax(self.dim_reduce, keepdim=not self.flatten)
        return torch.cat((x_avg, x_max), self.dim_cat)

# ---- SelectAdaptivePool2d (target) ----
class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHW',
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        pool_type = pool_type.lower()
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            assert output_size == 1, 'Fast pooling and non NCHW input formats require output_size == 1.'
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type == 'fast' or pool_type.endswith('avg'):
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            else:
                assert False, 'Invalid pool type: %s' % pool_type
            self.flatten = nn.Identity()
        else:
            assert input_fmt == 'NCHW'
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool2d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool2d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            elif pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            else:
                assert False, 'Invalid pool type: %s' % pool_type
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'

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
        self.select_adaptive_pool = SelectAdaptivePool2d(output_size=1, pool_type='avg', flatten=True, input_fmt='NCHW')
        self.classifier = nn.Linear(64, self.num_classes)

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
        x = self.select_adaptive_pool(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
