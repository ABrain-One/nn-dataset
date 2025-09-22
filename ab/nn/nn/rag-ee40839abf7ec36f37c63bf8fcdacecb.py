# Auto-generated single-file for ConvLayer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- timm.models.tiny_vit.ConvNorm ----
class ConvNorm(torch.nn.Sequential):
    def __init__(self, in_chs, out_chs, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.conv.groups, w.size(0), w.shape[2:],
            stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

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

# ---- timm.models.tiny_vit.MBConv ----
class MBConv(nn.Module):
    def __init__(self, in_chs, out_chs, expand_ratio, act_layer, drop_path):
        super().__init__()
        mid_chs = int(in_chs * expand_ratio)
        self.conv1 = ConvNorm(in_chs, mid_chs, ks=1)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(mid_chs, mid_chs, ks=3, stride=1, pad=1, groups=mid_chs)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(mid_chs, out_chs, ks=1, bn_weight_init=0.0)
        self.act3 = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x

# ---- ConvLayer (target) ----
class ConvLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            act_layer,
            drop_path=0.,
            conv_expand_ratio=4.,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.Sequential(*[
            MBConv(
                dim, dim, conv_expand_ratio, act_layer,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvLayer(dim=32, depth=2, act_layer=nn.ReLU, drop_path=0.1, conv_expand_ratio=2.0),
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
