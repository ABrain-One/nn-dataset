# Auto-generated single-file for NextConvBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import collections
from itertools import repeat
from collections import *

# ---- timm.models.nextvit.ConvAttention ----
class ConvAttention(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_chs, head_dim, norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU):
        super(ConvAttention, self).__init__()
        self.group_conv3x3 = nn.Conv2d(
            out_chs, out_chs,
            kernel_size=3, stride=1, padding=1, groups=out_chs // head_dim, bias=False
        )
        self.norm = norm_layer(out_chs)
        self.act = act_layer()
        self.projection = nn.Conv2d(out_chs, out_chs, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out

# ---- timm.models.nextvit.PatchEmbed ----
class PatchEmbed(nn.Module):
    def __init__(self,
            in_chs,
            out_chs,
            stride=1,
            norm_layer = nn.BatchNorm2d,
    ):
        super(PatchEmbed, self).__init__()

        if stride == 2:
            self.pool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_chs)
        elif in_chs != out_chs:
            self.pool = nn.Identity()
            self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_chs)
        else:
            self.pool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))

# ---- timm.layers.drop.drop_path ----
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

# ---- timm.layers.drop.DropPath ----
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

# ---- timm.layers.helpers._ntuple ----
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# ---- timm.models.nextvit.merge_pre_bn ----
def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_chs, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupported bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupported bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupported bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupported bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = (
                scale_invstd_2 * pre_bn_2.weight
                * (pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean)
                + pre_bn_2.bias
        )

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.layers.mlp.ConvMlp ----
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims (for 2D NCHW tensors)
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# ---- NextConvBlock (target) ----
class NextConvBlock(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            drop_path=0.,
            drop=0.,
            head_dim=32,
            mlp_ratio=3.,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU
    ):
        super(NextConvBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert out_chs % head_dim == 0

        self.patch_embed = PatchEmbed(in_chs, out_chs, stride, norm_layer=norm_layer)
        self.mhca = ConvAttention(
            out_chs,
            head_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.attn_drop_path = DropPath(drop_path)

        self.norm = norm_layer(out_chs)
        self.mlp = ConvMlp(
            out_chs,
            hidden_features=int(out_chs * mlp_ratio),
            drop=drop,
            bias=True,
            act_layer=act_layer,
        )
        self.mlp_drop_path = DropPath(drop_path)
        self.is_fused = False

    def reparameterize(self):
        if not self.is_fused:
            merge_pre_bn(self.mlp.fc1, self.norm)
            self.norm = nn.Identity()
            self.is_fused = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attn_drop_path(self.mhca(x))

        out = self.norm(x)
        x = x + self.mlp_drop_path(self.mlp(out))
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
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )
        self.next_conv_block = NextConvBlock(
            in_chs=32,
            out_chs=32,
            stride=1,
            drop_path=0.0,
            drop=0.0,
            head_dim=8,
            mlp_ratio=2.0,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU
        )
        self.classifier = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.next_conv_block(x)
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
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
