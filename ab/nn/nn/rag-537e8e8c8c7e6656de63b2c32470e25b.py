# Auto-generated single-file for CecaModule
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
import math
from typing import Type
from typing import Union
from typing import Optional
from typing import Callable

# ---- original imports from contributing modules ----

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

# ---- timm.layers.typing.LayerType ----
LayerType = Union[str, Callable, Type[torch.nn.Module]]

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

# ---- CecaModule (target) ----
class CecaModule(nn.Module):
    """Constructs a circular ECA module.

    ECA module where the conv uses circular padding rather than zero padding.
    Unlike the spatial dimension, the channels do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without significantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
        gamm: used in kernel_size calc, see above
        beta: used in kernel_size calc, see above
        act_layer: optional non-linearity after conv, enables conv bias, this is an experiment
        gate_layer: gating non-linearity to use
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid'):
        super(CecaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        has_act = act_layer is not None
        assert kernel_size % 2 == 1

        # PyTorch circular padding mode is buggy as of pytorch 1.4
        # see https://github.com/pytorch/pytorch/pull/17240
        # implement manual circular padding
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=has_act)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        # Manually implement circular padding, F.pad does not seemed to be bugged
        y = F.pad(y, (self.padding, self.padding), mode='circular')
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self.ceca = CecaModule(channels=32, kernel_size=3, gamma=2, beta=1)

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.ceca(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
