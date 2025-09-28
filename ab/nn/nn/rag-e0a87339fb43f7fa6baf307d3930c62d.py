# Auto-generated single-file for ReductionCell0
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# in_ch, out_ch
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import math
import collections
from itertools import repeat
from typing import Type
from functools import partial
from typing import List
from typing import Union
from typing import Optional
from collections import *
from typing import Tuple

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.layers.padding.get_same_padding ----
def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    if isinstance(x, torch.Tensor):
        return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
    else:
        return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

# ---- timm.layers.padding.pad_same ----
def pad_same(
        x,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
        value: float = 0,
):
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    return x

# ---- timm.layers.cond_conv2d.get_condconv_initializer ----
def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = math.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer

# ---- timm.layers.conv2d_same.conv2d_same ----
def conv2d_same(
        x,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

# ---- timm.layers.padding.pad_same_arg ----
def pad_same_arg(
        input_size: List[int],
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
) -> List[int]:
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = get_same_padding(ih, kh, stride[0], dilation[0])
    pad_w = get_same_padding(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

# ---- timm.layers.conv2d_same.Conv2dSameExport ----
class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions

    NOTE: This does not currently work with torch.jit.script
    """

    # pylint: disable=unused-argument
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, 0, dilation, groups, bias,
        )
        self.pad = None
        self.pad_input_size = (0, 0)

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = pad_same_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        x = self.pad(x)
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

# ---- timm.layers.mixed_conv2d._split_channels ----
def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.layers.padding.get_padding ----
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> Union[int, List[int]]:
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

# ---- timm.layers.padding.is_static_pad ----
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return all([is_static_pad(*a) for a in zip(kernel_size, stride, dilation)])
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

# ---- timm.layers.padding.get_padding_value ----
def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

# ---- timm.layers._fx._leaf_modules ----
_leaf_modules = set()

# ---- timm.layers._fx.register_notrace_module ----
def register_notrace_module(module: Type[nn.Module]):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module

# ---- timm.layers.pool2d_same.AvgPool2dSame ----
class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)

# ---- timm.layers.pool2d_same.MaxPool2dSame ----
class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float('inf'))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)

# ---- timm.layers.cond_conv2d.CondConv2d ----
class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = math.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        # reshape instead of view to work with channels_last input
        x = x.reshape(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out

# ---- timm.layers.conv2d_same.Conv2dSame ----
class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, 0, dilation, groups, bias,
        )

    def forward(self, x):
        return conv2d_same(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

# ---- timm.layers.pool2d_same.create_pool2d ----
def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    else:
        if pool_type == 'avg':
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == 'max':
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'

# ---- timm.layers.config._EXPORTABLE ----
_EXPORTABLE = False

# ---- timm.layers.config.is_exportable ----
def is_exportable():
    return _EXPORTABLE

# ---- timm.layers.config._SCRIPTABLE ----
_SCRIPTABLE = False

# ---- timm.layers.config.is_scriptable ----
def is_scriptable():
    return _SCRIPTABLE

# ---- timm.layers.conv2d_same._USE_EXPORT_CONV ----
_USE_EXPORT_CONV = False

# ---- timm.layers.conv2d_same.create_conv2d_pad ----
def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if _USE_EXPORT_CONV and is_exportable():
            # older PyTorch ver needed this to export same padding reasonably
            assert not is_scriptable()  # Conv2DSameExport does not work with jit
            return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        else:
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

# ---- timm.layers.mixed_conv2d.MixedConv2d ----
class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x

# ---- timm.layers.create_conv2d.create_conv2d ----
def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        if 'groups' in kwargs:
            groups = kwargs.pop('groups')
            if groups == in_channels:
                kwargs['depthwise'] = True
            else:
                assert groups == 1
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m

# ---- timm.models.nasnet.ActConvBn ----
class ActConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=''):
        super(ActConvBn, self).__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

# ---- timm.models.nasnet.SeparableConv2d ----
class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=''):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = create_conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv2d = create_conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x

# ---- timm.models.nasnet.BranchSeparables ----
class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_type='', stem_cell=False):
        super(BranchSeparables, self).__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, middle_channels, kernel_size, stride=stride, padding=pad_type)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001, momentum=0.1)
        self.act_2 = nn.ReLU(inplace=True)
        self.separable_2 = SeparableConv2d(
            middle_channels, out_channels, kernel_size, stride=1, padding=pad_type)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x

# ---- ReductionCell0 (target) ----
class ReductionCell0(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)

        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)

        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)

        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)

        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)

        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out

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
        self.reduction_cell = ReductionCell0(
            in_chs_left=32, out_chs_left=32,
            in_chs_right=32, out_chs_right=32
        )
        self.classifier = nn.Linear(128, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x_prev = x
        x = self.reduction_cell(x, x_prev)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
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
