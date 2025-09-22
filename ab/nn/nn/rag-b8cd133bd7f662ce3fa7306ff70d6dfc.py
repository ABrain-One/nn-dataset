# Auto-generated single-file for ClassifierHead
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from enum import Enum
from typing import Union
from typing import Optional
from typing import Tuple

# ---- timm.layers.format.Format ----
class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

# ---- timm.models.hgnet.LearnableAffineBlock ----
class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias

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

# ---- timm.layers.adaptive_avgmax_pool.SelectAdaptivePool2d ----
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

# ---- ClassifierHead (target) ----
class ClassifierHead(nn.Module):
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            hidden_size: Optional[int] = 2048,
            use_lab: bool = False
    ):
        super(ClassifierHead, self).__init__()
        self.num_features = in_features
        if pool_type is not None:
            if not pool_type:
                assert num_classes == 0, 'Classifier head must be removed if pooling is disabled'

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        if hidden_size is not None:
            self.num_features = hidden_size
            last_conv = nn.Conv2d(
                in_features,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            act = nn.ReLU()
            if use_lab:
                lab = LearnableAffineBlock()
                self.last_conv = nn.Sequential(last_conv, act, lab)
            else:
                self.last_conv = nn.Sequential(last_conv, act)
        else:
            self.last_conv = nn.Identity()

        self.dropout = nn.Dropout(drop_rate)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()  # don't flatten if pooling disabled
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            if not pool_type:
                assert num_classes == 0, 'Classifier head must be removed if pooling is disabled'
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.flatten = nn.Flatten(1) if pool_type else nn.Identity()  # don't flatten if pooling disabled

        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.last_conv(x)
        x = self.dropout(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        x = self.fc(x)
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
        self.classifier_head = ClassifierHead(
            in_features=32,
            num_classes=self.num_classes,
            pool_type='avg',
            drop_rate=0.1,
            hidden_size=None,
            use_lab=False
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
        x = self.classifier_head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
