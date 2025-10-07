# Auto-generated single-file for RepMixer
# Dependencies are emitted in topological order (utilities first).
from collections.abc import Sequence

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import math
import collections
from itertools import repeat
from typing import Type
from functools import partial
def _assert(condition, message): assert condition, message
from typing import Dict
def handle_torch_function(*args, **kwargs): pass
from typing import List
import numbers
from typing import Union
from typing import Optional
from functools import *
from collections import *
from math import comb
from typing import Any
from torch.nn.functional import layer_norm as fused_rms_norm
class InPlaceABN: pass
inplace_abn = InPlaceABN
from typing import Callable
from torch.nn.functional import layer_norm as fused_layer_norm_affine
from typing import Tuple
from torch.nn.functional import layer_norm as fused_rms_norm_affine
import types
import functools
def has_torch_function_variadic(*args, **kwargs): return False

# ---- timm.models.fastvit.LayerScale2d ----
class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

# ---- timm.layers.activations.GELU ----
class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)

# ---- timm.layers.activations.GELUTanh ----
class GELUTanh(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(GELUTanh, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input, approximate='tanh')

# ---- timm.layers.activations.PReLU ----
class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.weight)

# ---- timm.layers.activations.Sigmoid ----
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()

# ---- timm.layers.activations.Tanh ----
class Tanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()

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

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

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

# ---- timm.layers.evo_norm.instance_std ----
def instance_std(x, eps: float = 1e-5):
    std = x.float().var(dim=(2, 3), unbiased=False, keepdim=True).add(eps).sqrt().to(x.dtype)
    return std.expand(x.shape)

# ---- timm.layers.evo_norm.EvoNorm2dB0 ----
class EvoNorm2dB0(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-3, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.v is not None:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                # var = manual_var(x, dim=(0, 2, 3)).squeeze()
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach() * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            left = var.add(self.eps).sqrt_().to(x_dtype).view(v_shape).expand_as(x)
            v = self.v.to(x_dtype).view(v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.to(x_dtype).view(v_shape) + self.bias.to(x_dtype).view(v_shape)

# ---- timm.layers.evo_norm.instance_rms ----
def instance_rms(x, eps: float = 1e-5):
    rms = x.float().square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt().to(x.dtype)
    return rms.expand(x.shape)

# ---- timm.layers.evo_norm.EvoNorm2dB1 ----
class EvoNorm2dB1(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach().to(self.running_var.dtype) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = var.to(x_dtype).view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = (x + 1) * instance_rms(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.evo_norm.EvoNorm2dB2 ----
class EvoNorm2dB2(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach().to(self.running_var.dtype) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = var.to(x_dtype).view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = instance_rms(x, self.eps) - x
            x = x / left.max(right)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.evo_norm.group_std ----
def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
    B, C, H, W = x.shape
    x_dtype = x.dtype
    _assert(C % groups == 0, '')
    if flatten:
        x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
        std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
    return std.expand(x.shape).reshape(B, C, H, W)

# ---- timm.layers.evo_norm.EvoNorm2dS0 ----
class EvoNorm2dS0(nn.Module):
    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.v is not None:
            v = self.v.view(v_shape).to(x_dtype)
            x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.evo_norm.EvoNorm2dS0a ----
class EvoNorm2dS0a(EvoNorm2dS0):
    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        d = group_std(x, self.groups, self.eps)
        if self.v is not None:
            v = self.v.view(v_shape).to(x_dtype)
            x = x * (x * v).sigmoid()
        x = x / d
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.activations.hard_mish ----
def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)

# ---- timm.layers.activations.HardMish ----
class HardMish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)

# ---- timm.layers.activations.hard_sigmoid ----
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

# ---- timm.layers.activations.HardSigmoid ----
class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)

# ---- timm.layers.activations.hard_swish ----
def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)

# ---- timm.layers.activations.HardSwish ----
class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)

# ---- timm.layers.activations.mish ----
def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())

# ---- timm.layers.activations.Mish ----
class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)

# ---- timm.layers.activations.quick_gelu ----
def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)

# ---- timm.layers.activations.QuickGELU ----
class QuickGELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(QuickGELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quick_gelu(input)

# ---- timm.layers.activations.swish ----
def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

# ---- timm.layers.activations.Swish ----
class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

# ---- timm.layers.activations_me.hard_mish_bwd ----
def hard_mish_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= -2.)
    m = torch.where((x >= -2.) & (x <= 0.), x + 1., m)
    return grad_output * m

# ---- timm.layers.activations_me.hard_mish_fwd ----
def hard_mish_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)

# ---- timm.layers.activations_me.HardMishAutoFn ----
class HardMishAutoFn(torch.autograd.Function):
    """ A memory efficient variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_fwd(x)

    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_bwd(x, grad_output)

# ---- timm.layers.activations_me.HardMishMe ----
class HardMishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishAutoFn.apply(x)

# ---- timm.layers.activations_me.hard_sigmoid_bwd ----
def hard_sigmoid_bwd(x, grad_output):
    m = torch.ones_like(x) * ((x >= -3.) & (x <= 3.)) / 6.
    return grad_output * m

# ---- timm.layers.activations_me.hard_sigmoid_fwd ----
def hard_sigmoid_fwd(x, inplace: bool = False):
    return (x + 3).clamp(min=0, max=6).div(6.)

# ---- timm.layers.activations_me.HardSigmoidAutoFn ----
class HardSigmoidAutoFn(torch.autograd.Function):
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_fwd(x)

    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_bwd(x, grad_output)

# ---- timm.layers.activations_me.HardSigmoidMe ----
class HardSigmoidMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidMe, self).__init__()

    def forward(self, x):
        return HardSigmoidAutoFn.apply(x)

# ---- timm.layers.activations_me.hard_swish_bwd ----
def hard_swish_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= 3.)
    m = torch.where((x >= -3.) & (x <= 3.),  x / 3. + .5, m)
    return grad_output * m

# ---- timm.layers.activations_me.hard_swish_fwd ----
def hard_swish_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.)

# ---- timm.layers.activations_me.HardSwishAutoFn ----
class HardSwishAutoFn(torch.autograd.Function):
    """A memory efficient HardSwish activation"""
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_fwd(x)

    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_bwd(x, grad_output)

    def symbolic(g, self):
        input = g.op("Add", self, g.op('Constant', value_t=torch.tensor(3, dtype=torch.float)))
        hardtanh_ = g.op("Clip", input, g.op('Constant', value_t=torch.tensor(0, dtype=torch.float)), g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        hardtanh_ = g.op("Div", hardtanh_, g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        return g.op("Mul", self, hardtanh_)

# ---- timm.layers.activations_me.HardSwishMe ----
class HardSwishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishMe, self).__init__()

    def forward(self, x):
        return HardSwishAutoFn.apply(x)

# ---- timm.layers.activations_me.mish_bwd ----
def mish_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

# ---- timm.layers.activations_me.mish_fwd ----
def mish_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))

# ---- timm.layers.activations_me.MishAutoFn ----
class MishAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient variant of Mish
    """
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_fwd(x)

    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_bwd(x, grad_output)

# ---- timm.layers.activations_me.MishMe ----
class MishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishAutoFn.apply(x)

# ---- timm.layers.activations_me.swish_bwd ----
def swish_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))

# ---- timm.layers.activations_me.swish_fwd ----
def swish_fwd(x):
    return x.mul(torch.sigmoid(x))

# ---- timm.layers.activations_me.SwishAutoFn ----
class SwishAutoFn(torch.autograd.Function):
    """ optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    def symbolic(g, x):
        return g.op("Mul", x, g.op("Sigmoid", x))

    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_bwd(x, grad_output)

# ---- timm.layers.activations_me.SwishMe ----
class SwishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishMe, self).__init__()

    def forward(self, x):
        return SwishAutoFn.apply(x)

# ---- timm.layers.evo_norm.group_rms ----
def group_rms(x, groups: int = 32, eps: float = 1e-5):
    B, C, H, W = x.shape
    _assert(C % groups == 0, '')
    x_dtype = x.dtype
    x = x.reshape(B, groups, C // groups, H, W)
    rms = x.float().square().mean(dim=(2, 3, 4), keepdim=True).add(eps).sqrt_().to(x_dtype)
    return rms.expand(x.shape).reshape(B, C, H, W)

# ---- timm.layers.filter_response_norm.inv_instance_rms ----
def inv_instance_rms(x, eps: float = 1e-5):
    rms = x.square().float().mean(dim=(2, 3), keepdim=True).add(eps).rsqrt().to(x.dtype)
    return rms.expand(x.shape)

# ---- timm.layers.filter_response_norm.FilterResponseNormTlu2d ----
class FilterResponseNormTlu2d(nn.Module):
    def __init__(self, num_features, apply_act=True, eps=1e-5, rms=True, **_):
        super(FilterResponseNormTlu2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.tau = nn.Parameter(torch.zeros(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)
        return torch.maximum(x, self.tau.reshape(v_shape).to(dtype=x_dtype)) if self.tau is not None else x

# ---- timm.layers.fast_norm.get_autocast_dtype ----
def get_autocast_dtype(device: str = 'cuda'):
    try:
        return torch.get_autocast_dtype(device)
    except (AttributeError, TypeError):
        # dispatch to older device specific fns, only covering cuda/cpu devices here
        if device == 'cpu':
            return torch.get_autocast_cpu_dtype()
        else:
            assert device == 'cuda'
            return torch.get_autocast_gpu_dtype()

# ---- timm.layers.fast_norm.is_autocast_enabled ----
def is_autocast_enabled(device: str = 'cuda'):
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:
        # dispatch to older device specific fns, only covering cuda/cpu devices here
        if device == 'cpu':
            return torch.is_autocast_cpu_enabled()
        else:
            assert device == 'cuda'
            return torch.is_autocast_enabled()  # defaults cuda (only cuda on older pytorch)

# ---- timm.layers.fast_norm.fast_group_norm ----
def fast_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.group_norm(x, num_groups, weight, bias, eps)

    if is_autocast_enabled(x.device.type):
        # normally native AMP casts GN inputs to float32
        # here we use the low precision autocast dtype
        dt = get_autocast_dtype(x.device.type)
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt) if bias is not None else None

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        return F.group_norm(x, num_groups, weight, bias, eps)

# ---- timm.layers.norm_act._num_groups ----
def _num_groups(num_channels: int, num_groups: int, group_size: int):
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups

# ---- timm.layers.fast_norm.rms_norm ----
def rms_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    norm_ndim = len(normalized_shape)
    v = x.pow(2)
    if torch.jit.is_scripting():
        # ndim = len(x.shape)
        # dims = list(range(ndim - norm_ndim, ndim))  # this doesn't work on pytorch <= 1.13.x
        # NOTE -ve dims cause torchscript to crash in some cases, out of options to work around
        assert norm_ndim == 1
        v = torch.mean(v, dim=-1).unsqueeze(-1)  # ts crashes with -ve dim + keepdim=True
    else:
        dims = tuple(range(-1, -norm_ndim - 1, -1))
        v = torch.mean(v, dim=dims, keepdim=True)
    x = x * torch.rsqrt(v + eps)
    if weight is not None:
        x = x * weight
    return x

# ---- timm.layers.fast_norm.rms_norm2d ----
def rms_norm2d(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    assert len(normalized_shape) == 1
    v = x.pow(2)
    v = torch.mean(v, dim=1, keepdim=True)
    x = x * torch.rsqrt(v + eps)
    if weight is not None:
        x = x * weight.reshape(1, -1, 1, 1)
    return x

# ---- timm.layers.helpers.make_divisible ----
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

# ---- timm.models.fastvit.num_groups ----
def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size

# ---- timm.layers._fx._leaf_modules ----
_leaf_modules = set()

# ---- timm.layers._fx.register_notrace_module ----
def register_notrace_module(module: Type[nn.Module]):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module

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

# ---- timm.layers.inplace_abn.InplaceAbn ----
class InplaceAbn(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, apply_act=True,
                 act_layer="leaky_relu", act_param=0.01, drop_layer=None):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity', '')
                self.act_name = act_layer if act_layer else 'identity'
            else:
                # convert act layer passed as type to string
                if act_layer == nn.ELU:
                    self.act_name = 'elu'
                elif act_layer == nn.LeakyReLU:
                    self.act_name = 'leaky_relu'
                elif act_layer is None or act_layer == nn.Identity:
                    self.act_name = 'identity'
                else:
                    assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        output = inplace_abn(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps, self.act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.layers.padding.get_padding ----
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> Union[int, List[int]]:
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride, dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

# ---- timm.layers.blur_pool.BlurPool2d ----
class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(
            self,
            channels: Optional[int] = None,
            filt_size: int = 3,
            stride: int = 2,
            pad_mode: str = 'reflect',
    ) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4

        # (0.5 + 0.5 x)^N => coefficients = C(N,k) / 2^N,  k = 0..N
        coeffs = torch.tensor(
            [comb(filt_size - 1, k) for k in range(filt_size)],
            dtype=torch.float32,
        ) / (2 ** (filt_size - 1))  # normalise so coefficients sum to 1
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        if channels is not None:
            blur_filter = blur_filter.repeat(self.channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, mode=self.pad_mode)
        if self.channels is None:
            channels = x.shape[1]
            weight = self.filt.expand(channels, 1, self.filt_size, self.filt_size)
        else:
            channels = self.channels
            weight = self.filt
        return F.conv2d(x, weight, stride=self.stride, groups=channels)

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

# ---- timm.layers.typing.LayerType ----
LayerType = Union[str, Callable, Type[torch.nn.Module]]

# ---- timm.layers.blur_pool.create_aa ----
def create_aa(
        aa_layer: LayerType,
        channels: Optional[int] = None,
        stride: int = 2,
        enable: bool = True,
        noop: Optional[Type[nn.Module]] = nn.Identity
) -> nn.Module:
    """ Anti-aliasing """
    if not aa_layer or not enable:
        return noop() if noop is not None else None

    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace('_', '').replace('-', '')
        if aa_layer == 'avg' or aa_layer == 'avgpool':
            aa_layer = nn.AvgPool2d
        elif aa_layer == 'blur' or aa_layer == 'blurpool':
            aa_layer = BlurPool2d
        elif aa_layer == 'blurpc':
            aa_layer = partial(BlurPool2d, pad_mode='constant')

        else:
            assert False, f"Unknown anti-aliasing layer ({aa_layer})."

    try:
        return aa_layer(channels=channels, stride=stride)
    except TypeError as e:
        return aa_layer(stride)

# ---- timm.layers.create_act._has_hardsigmoid ----
_has_hardsigmoid = 'hardsigmoid' in dir(torch.nn.functional)

# ---- timm.layers.create_act._has_hardswish ----
_has_hardswish = 'hardswish' in dir(torch.nn.functional)

# ---- timm.layers.create_act._has_mish ----
_has_mish = 'mish' in dir(torch.nn.functional)

# ---- timm.layers.create_act._has_silu ----
_has_silu = 'silu' in dir(torch.nn.functional)

# ---- timm.layers.create_act._ACT_LAYER_DEFAULT ----
_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=nn.Mish if _has_mish else Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoid,
    hard_swish=nn.Hardswish if _has_hardswish else HardSwish,
    hard_mish=HardMish,
    identity=nn.Identity,
)

# ---- timm.layers.create_act._ACT_LAYER_ME ----
_ACT_LAYER_ME = dict(
    silu=nn.SiLU if _has_silu else SwishMe,
    swish=nn.SiLU if _has_silu else SwishMe,
    mish=nn.Mish if _has_mish else MishMe,
    hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoidMe,
    hard_swish=nn.Hardswish if _has_hardswish else HardSwishMe,
    hard_mish=HardMishMe,
)

# ---- timm.layers.create_act.get_act_layer ----
def get_act_layer(name: Optional[LayerType] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not name:
        return None
    name = name.lower()
    if not (is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    return _ACT_LAYER_DEFAULT[name]

# ---- timm.layers.create_act.create_act_layer ----
def create_act_layer(
        name: Optional[LayerType],
        inplace: Optional[bool] = None,
        **kwargs
):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        # recover if act layer doesn't have inplace arg
        return act_layer(**kwargs)

# ---- timm.layers.evo_norm.EvoNorm2dS1 ----
class EvoNorm2dS1(nn.Module):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-5, **_):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        self.apply_act = apply_act  # apply activation (non-linearity)
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.pre_act_norm = False
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.evo_norm.EvoNorm2dS1a ----
class EvoNorm2dS1a(EvoNorm2dS1):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.evo_norm.EvoNorm2dS2 ----
class EvoNorm2dS2(nn.Module):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-5, **_):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        self.apply_act = apply_act  # apply activation (non-linearity)
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.apply_act:
            x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.evo_norm.EvoNorm2dS2a ----
class EvoNorm2dS2a(EvoNorm2dS2):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape).to(x_dtype) + self.bias.view(v_shape).to(x_dtype)

# ---- timm.layers.filter_response_norm.FilterResponseNormAct2d ----
class FilterResponseNormAct2d(nn.Module):
    def __init__(self, num_features, apply_act=True, act_layer=nn.ReLU, inplace=None, rms=True, eps=1e-5, **_):
        super(FilterResponseNormAct2d, self).__init__()
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer, inplace=inplace)
        else:
            self.act = nn.Identity()
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)
        return self.act(x)

# ---- timm.layers.squeeze_excite.SEModule ----
class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

# ---- timm.layers.norm_act._create_act ----
def _create_act(
        act_layer: LayerType,
        act_kwargs: Dict[str, Any] = None,
        inplace: Optional[bool] = False,
        apply_act: bool = True,
) -> nn.Module:
    act_kwargs = act_kwargs or {}
    act_kwargs.setdefault('inplace', inplace)
    act = None
    if apply_act:
        act = create_act_layer(act_layer, **act_kwargs)
    return nn.Identity() if act is None else act

# ---- timm.layers.norm_act.BatchNormAct2d ----
class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.norm_act.LayerNormAct2dFp32 ----
class LayerNormAct2dFp32(nn.LayerNorm):

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        x = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.norm_act.LayerNormActFp32 ----
class LayerNormActFp32(nn.LayerNorm):

    def __init__(
            self,
            normalization_shape: Union[int, List[int], torch.Size],
            eps: float = 1e-5,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            **kwargs,
    ):
        super().__init__(normalization_shape, eps=eps, elementwise_affine=affine, **kwargs)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        x = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.fast_norm._USE_FAST_NORM ----
_USE_FAST_NORM = False  # defaulting to False for now

# ---- timm.layers.fast_norm.is_fast_norm ----
def is_fast_norm():
    return _USE_FAST_NORM

# ---- timm.layers.norm_act.GroupNorm1Act ----
class GroupNorm1Act(nn.GroupNorm):
    _fast_norm: torch.jit.Final[bool]

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
    ):
        super(GroupNorm1Act, self).__init__(1, num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.norm_act.GroupNormAct ----
class GroupNormAct(nn.GroupNorm):
    _fast_norm: torch.jit.Final[bool]

    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(
            self,
            num_channels: int,
            num_groups: int = 32,
            eps: float = 1e-5,
            affine: bool = True,
            group_size: Optional[int] = None,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
    ):
        super(GroupNormAct, self).__init__(
            _num_groups(num_channels, num_groups, group_size),
            num_channels,
            eps=eps,
            affine=affine,
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.fast_norm.has_apex ----
try:
    from apex.normalization.fused_layer_norm import fused_layer_norm_affine
    has_apex = True
except ImportError:
    has_apex = False

# ---- timm.layers.fast_norm.fast_layer_norm ----
def fast_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    if has_apex:
        return fused_layer_norm_affine(x, weight, bias, normalized_shape, eps)

    if is_autocast_enabled(x.device.type):
        # normally native AMP casts LN inputs to float32
        # apex LN does not, this is behaving like Apex
        dt = get_autocast_dtype(x.device.type)
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt) if bias is not None else None

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

# ---- timm.layers.norm_act.LayerNormAct ----
class LayerNormAct(nn.LayerNorm):
    _fast_norm: torch.jit.Final[bool]

    def __init__(
            self,
            normalization_shape: Union[int, List[int], torch.Size],
            eps: float = 1e-5,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            **kwargs,
    ):
        super(LayerNormAct, self).__init__(normalization_shape, eps=eps, elementwise_affine=affine, **kwargs)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.norm_act.LayerNormAct2d ----
class LayerNormAct2d(nn.LayerNorm):
    _fast_norm: torch.jit.Final[bool]

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.fast_norm.has_apex_rmsnorm ----
try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine, fused_rms_norm
    has_apex_rmsnorm = True
except ImportError:
    has_apex_rmsnorm = False

# ---- timm.layers.fast_norm.fast_rms_norm2d ----
def fast_rms_norm2d(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # this must be by itself, cannot merge with has_apex_rmsnorm
        return rms_norm2d(x, normalized_shape, weight, eps)

    if has_apex_rmsnorm:
        x = x.permute(0, 2, 3, 1)
        if weight is None:
            x = fused_rms_norm(x, normalized_shape, eps)
        else:
            x = fused_rms_norm_affine(x, weight, normalized_shape, eps)
        x = x.permute(0, 3, 1, 2)

    if is_autocast_enabled(x.device.type):
        # normally native AMP casts norm inputs to float32 and leaves the output as float32
        # apex does not, this is behaving like Apex
        dt = get_autocast_dtype(x.device.type)
        x, weight = x.to(dt), weight.to(dt)

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x = rms_norm2d(x, normalized_shape, weight, eps)

    return x

# ---- timm.layers.norm.RmsNorm2d ----
class RmsNorm2d(nn.Module):
    """ RmsNorm2D for NCHW tensors, w/ fast apex or cast norm if available

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always use APEX RmsNorm if is installed.
        if self._fast_norm:
            x = fast_rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        return x

# ---- timm.layers.norm_act.RmsNormAct2d ----
class RmsNormAct2d(RmsNorm2d):
    """ RMSNorm + Activation for '2D' NCHW tensors

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__(channels=num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)
        self._fast_norm = is_fast_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.norm_act.RmsNormAct2dFp32 ----
class RmsNormAct2dFp32(RmsNorm2d):
    """ RMSNorm + Activation for '2D' NCHW tensors

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__(channels=num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        x = rms_norm2d(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.fast_norm.has_torch_rms_norm ----
has_torch_rms_norm = hasattr(F, 'rms_norm')

# ---- timm.layers.fast_norm.fast_rms_norm ----
def fast_rms_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # this must be by itself, cannot merge with has_apex_rmsnorm
        return rms_norm(x, normalized_shape, weight, eps)

    if has_apex_rmsnorm:
        if weight is None:
            return fused_rms_norm(x, normalized_shape, eps)
        else:
            return fused_rms_norm_affine(x, weight, normalized_shape, eps)

    if is_autocast_enabled(x.device.type):
        # normally native AMP casts LN inputs to float32 and leaves the output as float32
        # apex LN does not, this is behaving like Apex
        dt = get_autocast_dtype(x.device.type)
        x, weight = x.to(dt), weight.to(dt)

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        if has_torch_rms_norm:
            x = F.rms_norm(x, normalized_shape, weight, eps)
        else:
            x = rms_norm(x, normalized_shape, weight, eps)

    return x

# ---- timm.layers.norm.RmsNorm ----
class RmsNorm(nn.Module):
    """ RmsNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always uses APEX RmsNorm if installed.
        if self._fast_norm:
            x = fast_rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return x

# ---- timm.layers.norm_act.RmsNormAct ----
class RmsNormAct(RmsNorm):
    """ RMSNorm + Activation for '2D' NCHW tensors

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            **kwargs,
    ):
        super().__init__(channels=num_channels, eps=eps, affine=affine, **kwargs)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)
        self._fast_norm = is_fast_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm(x, self.normalized_shape, self.weight, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.norm_act.RmsNormActFp32 ----
class RmsNormActFp32(RmsNorm):
    """ RMSNorm + Activation for '2D' NCHW tensors

    NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
    on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
    like https://github.com/pytorch/pytorch/pull/150576 lands.
    """
    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            apply_act: bool = True,
            act_layer: LayerType = nn.ReLU,
            act_kwargs: Dict[str, Any] = None,
            inplace: bool = True,
            drop_layer: Optional[Type[nn.Module]] = None,
            **kwargs,
    ):
        super().__init__(channels=num_channels, eps=eps, affine=affine, **kwargs)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        x = rms_norm(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)
        x = self.drop(x)
        x = self.act(x)
        return x

# ---- timm.layers.create_norm_act._NORM_ACT_MAP ----
_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    groupnorm1=GroupNorm1Act,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
    layernormfp32=LayerNormActFp32,
    layernorm2dfp32=LayerNormAct2dFp32,
    evonormb0=EvoNorm2dB0,
    evonormb1=EvoNorm2dB1,
    evonormb2=EvoNorm2dB2,
    evonorms0=EvoNorm2dS0,
    evonorms0a=EvoNorm2dS0a,
    evonorms1=EvoNorm2dS1,
    evonorms1a=EvoNorm2dS1a,
    evonorms2=EvoNorm2dS2,
    evonorms2a=EvoNorm2dS2a,
    frn=FilterResponseNormAct2d,
    frntlu=FilterResponseNormTlu2d,
    inplaceabn=InplaceAbn,
    iabn=InplaceAbn,
    rmsnorm=RmsNormAct,
    rmsnorm2d=RmsNormAct2d,
    rmsnormfp32=RmsNormActFp32,
    rmsnorm2dfp32=RmsNormAct2dFp32,
)

# ---- timm.layers.create_norm_act._NORM_ACT_REQUIRES_ARG ----
_NORM_ACT_REQUIRES_ARG = {
    BatchNormAct2d,
    GroupNormAct,
    GroupNorm1Act,
    LayerNormAct,
    LayerNormAct2d,
    LayerNormActFp32,
    LayerNormAct2dFp32,
    FilterResponseNormAct2d,
    InplaceAbn,
    RmsNormAct,
    RmsNormAct2d,
    RmsNormActFp32,
    RmsNormAct2dFp32,
}

# ---- timm.layers.create_norm_act._NORM_ACT_TYPES ----
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}

# ---- timm.layers.create_norm_act._NORM_TO_NORM_ACT_MAP ----
_NORM_TO_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
    groupnorm=GroupNormAct,
    groupnorm1=GroupNorm1Act,
    layernorm=LayerNormAct,
    layernorm2d=LayerNormAct2d,
    layernormfp32=LayerNormActFp32,
    layernorm2dfp32=LayerNormAct2dFp32,
    rmsnorm=RmsNormAct,
    rmsnorm2d=RmsNormAct2d,
    rmsnormfp32=RmsNormActFp32,
    rmsnorm2dfp32=RmsNormAct2dFp32,
)

# ---- timm.layers.create_norm_act.get_norm_act_layer ----
def get_norm_act_layer(
        norm_layer: LayerType,
        act_layer: Optional[LayerType] = None,
):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP[layer_name]
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        # Use reverse map to find the corresponding norm+act layer
        type_name = norm_layer.__name__.lower()
        norm_act_layer = _NORM_TO_NORM_ACT_MAP.get(type_name, None)
        assert norm_act_layer is not None, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args

    return norm_act_layer

# ---- timm.layers.typing.PadType ----
PadType = Union[str, int, Tuple[int, int]]

# ---- timm.layers.conv_bn_act.ConvNormAct ----
class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: PadType = '',
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            apply_norm: bool = True,
            apply_act: bool = True,
            norm_layer: LayerType = nn.BatchNorm2d,
            act_layer: Optional[LayerType] = nn.ReLU,
            aa_layer: Optional[LayerType] = None,
            drop_layer: Optional[Type[nn.Module]] = None,
            conv_kwargs: Optional[Dict[str, Any]] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
            act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ConvNormAct, self).__init__()
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        use_aa = aa_layer is not None and stride > 1

        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1 if use_aa else stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **conv_kwargs,
        )

        if apply_norm:
            # NOTE for backwards compatibility with models that use separate norm and act layer definitions
            norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
            # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
            self.bn = norm_act_layer(
                out_channels,
                apply_act=apply_act,
                act_kwargs=act_kwargs,
                **norm_kwargs,
            )
        else:
            self.bn = nn.Sequential()
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
                self.bn.add_module('drop', drop_layer())

        self.aa = create_aa(aa_layer, out_channels, stride=stride, enable=use_aa, noop=None)

    def in_channels(self):
        return self.conv.in_channels

    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        aa = getattr(self, 'aa', None)
        if aa is not None:
            x = self.aa(x)
        return x

# ---- timm.layers.squeeze_excite.SqueezeExcite ----
SqueezeExcite = SEModule  # alias

# ---- timm.models.fastvit.MobileOneBlock ----
class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            inference_mode: bool = False,
            use_se: bool = False,
            use_act: bool = True,
            use_scale_branch: bool = True,
            num_conv_branches: int = 1,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_chs: Number of channels in the input.
            out_chs: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            dilation: Kernel dilation factor.
            group_size: Convolution group size.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = num_groups(group_size, in_chs)
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        self.se = SqueezeExcite(out_chs, rd_divisor=1) if use_se else nn.Identity()

        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.reparam_conv = None

            self.identity = (
                nn.BatchNorm2d(num_features=in_chs)
                if out_chs == in_chs and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                self.conv_kxk = nn.ModuleList([
                    ConvNormAct(
                        self.in_chs,
                        self.out_chs,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        groups=self.groups,
                        apply_act=False,
                    ) for _ in range(self.num_conv_branches)
                ])
            else:
                self.conv_kxk = None

            # Re-parameterizable scale branch
            self.conv_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.conv_scale = ConvNormAct(
                    self.in_chs,
                    self.out_chs,
                    kernel_size=1,
                    stride=self.stride,
                    groups=self.groups,
                    apply_act=False
                )

        self.act = act_layer() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.reparam_conv is not None:
            return self.act(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Identity branch output
        identity_out = 0
        if self.identity is not None:
            identity_out = self.identity(x)

        # Scale branch output
        scale_out = 0
        if self.conv_scale is not None:
            scale_out = self.conv_scale(x)

        # Other kxk conv branches
        out = scale_out + identity_out
        if self.conv_kxk is not None:
            for rc in self.conv_kxk:
                out += rc(x)

        return self.act(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = create_conv2d(
            in_channels=self.in_chs,
            out_channels=self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()

        self.__delattr__("conv_kxk")
        self.__delattr__("conv_scale")
        if hasattr(self, "identity"):
            self.__delattr__("identity")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.conv_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.conv_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.conv_kxk is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.conv_kxk[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_chs // self.groups
                kernel_value = torch.zeros(
                    (self.in_chs, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_chs):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

# ---- RepMixer (target) ----
class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
            self,
            dim,
            kernel_size=3,
            layer_scale_init_value=1e-5,
            inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.reparam_conv = None
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
            )
            if layer_scale_init_value is not None:
                self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
            else:
                self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
        else:
            x = x + self.layer_scale(self.mixer(x) - self.norm(x))
        return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if isinstance(self.layer_scale, LayerScale2d):
            w = self.mixer.id_tensor + self.layer_scale.gamma.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale.gamma) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = create_conv2d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        self.__delattr__("layer_scale")

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
        self.rep_mixer = RepMixer(dim=64, kernel_size=3, layer_scale_init_value=1e-5, inference_mode=False)
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
        x = self.rep_mixer(x)
        x = x.mean(dim=(2, 3))
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
