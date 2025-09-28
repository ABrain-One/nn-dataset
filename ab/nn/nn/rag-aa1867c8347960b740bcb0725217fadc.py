# Auto-generated single-file for RepVitDownsample
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable
from typing import Type
from typing import Union
from typing import Optional
from typing import Callable

# ---- timm.models.repvit.ConvNorm ----
class ConvNorm(nn.Sequential):
    def __init__(self, in_dim, out_dim, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_dim, out_dim, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_dim))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

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

# ---- timm.models.repvit.RepVggDw ----
class RepVggDw(nn.Module):
    def __init__(self, ed, kernel_size, legacy=False):
        super().__init__()
        self.conv = ConvNorm(ed, ed, kernel_size, 1, (kernel_size - 1) // 2, groups=ed)
        if legacy:
            self.conv1 = ConvNorm(ed, ed, 1, 1, 0, groups=ed)
            # Make torchscript happy.
            self.bn = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
            self.bn = nn.BatchNorm2d(ed)
        self.dim = ed
        self.legacy = legacy

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)

    def fuse(self):
        conv = self.conv.fuse()

        if self.legacy:
            conv1 = self.conv1.fuse()
        else:
            conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        if not self.legacy:
            bn = self.bn
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w = conv.weight * w[:, None, None, None]
            b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps) ** 0.5
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
        return conv

# ---- timm.models.repvit.RepVitMlp ----
class RepVitMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, act_layer):
        super().__init__()
        self.conv1 = ConvNorm(in_dim, hidden_dim, 1, 1, 0)
        self.act = act_layer()
        self.conv2 = ConvNorm(hidden_dim, in_dim, 1, 1, 0, bn_weight_init=0)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

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

# ---- timm.layers.helpers.make_divisible ----
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

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

# ---- timm.layers.squeeze_excite.SqueezeExcite ----
SqueezeExcite = SEModule  # alias

# ---- timm.models.repvit.RepViTBlock ----
class RepViTBlock(nn.Module):
    def __init__(self, in_dim, mlp_ratio, kernel_size, use_se, act_layer, legacy=False):
        super(RepViTBlock, self).__init__()

        self.token_mixer = RepVggDw(in_dim, kernel_size, legacy)
        self.se = SqueezeExcite(in_dim, 0.25) if use_se else nn.Identity()
        self.channel_mixer = RepVitMlp(in_dim, in_dim * mlp_ratio, act_layer)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.se(x)
        identity = x
        x = self.channel_mixer(x)
        return identity + x

# ---- RepVitDownsample (target) ----
class RepVitDownsample(nn.Module):
    def __init__(self, in_dim, mlp_ratio, out_dim, kernel_size, act_layer, legacy=False):
        super().__init__()
        self.pre_block = RepViTBlock(in_dim, mlp_ratio, kernel_size, use_se=False, act_layer=act_layer, legacy=legacy)
        self.spatial_downsample = ConvNorm(in_dim, in_dim, kernel_size, 2, (kernel_size - 1) // 2, groups=in_dim)
        self.channel_downsample = ConvNorm(in_dim, out_dim, 1, 1)
        self.ffn = RepVitMlp(out_dim, out_dim * mlp_ratio, act_layer)

    def forward(self, x):
        x = self.pre_block(x)
        x = self.spatial_downsample(x)
        x = self.channel_downsample(x)
        identity = x
        x = self.ffn(x)
        return x + identity

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
        self.rep_vit_downsample = RepVitDownsample(in_dim=64, mlp_ratio=4, out_dim=128, kernel_size=3, act_layer=GELU, legacy=False)
        self.classifier = nn.Linear(128, self.num_classes)

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
        x = self.rep_vit_downsample(x)
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
