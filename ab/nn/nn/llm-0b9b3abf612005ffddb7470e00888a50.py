import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import copy

class MODELS:

    @staticmethod
    def build(cfg):
        return None

    @staticmethod
    def switch_scope_and_registry(scope):
        return MODELS()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

class HorNetLayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise ValueError('data_format must be channels_last or channels_first')
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GlobalLocalFilter(nn.Module):

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        self.pre_norm = HorNetLayerNorm(dim, eps=1e-06, data_format='channels_first')
        self.post_norm = HorNetLayerNorm(dim, eps=1e-06, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class LayerScale(nn.Module):

    def __init__(self, dim: int, layer_scale_init_value: Union[float, torch.Tensor]=1e-05, inplace: bool=False, data_format: str='channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight

def drop_path(x: torch.Tensor, drop_prob: float=0.0, training: bool=False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob: float=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

def get_dwconv(dim, kernel_size, bias=True):
    return nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias, groups=dim)

class gnConv(nn.Module):

    def __init__(self, dim, order=5, dw_cfg=dict(type='DW', kernel_size=7), scale=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        cfg = copy.deepcopy(dw_cfg)
        dw_type = cfg.pop('type')
        assert dw_type in ['DW', 'GF'], 'dw_type should be `DW` or `GF`'
        if dw_type == 'DW':
            self.dwconv = get_dwconv(sum(self.dims), **cfg)
        elif dw_type == 'GF':
            self.dwconv = GlobalLocalFilter(sum(self.dims), **cfg)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.projs = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])
        self.scale = scale

    def forward(self, x):
        x = self.proj_in(x)
        y, x = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)
        x = self.dwconv(x) * self.scale
        dw_list = torch.split(x, self.dims, dim=1)
        x = y * dw_list[0]
        for i in range(self.order - 1):
            x = self.projs[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x

class HorNetBlock(nn.Module):

    def __init__(self, dim, order=5, dw_cfg=dict(type='DW', kernel_size=7), scale=1.0, drop_path_rate=0.0, use_layer_scale=True):
        super().__init__()
        self.out_channels = dim
        self.norm1 = HorNetLayerNorm(dim, eps=1e-06, data_format='channels_first')
        self.gnconv = gnConv(dim, order, dw_cfg, scale)
        self.norm2 = HorNetLayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if use_layer_scale:
            self.gamma1 = LayerScale(dim, data_format='channels_first')
            self.gamma2 = LayerScale(dim)
        else:
            self.gamma1, self.gamma2 = (nn.Identity(), nn.Identity())
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.gamma1(self.gnconv(self.norm1(x))))
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
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
        self.hornet_block = HorNetBlock(dim=64, order=3, dw_cfg=dict(type='DW', kernel_size=7), scale=1.0, drop_path_rate=0.0, use_layer_scale=True)
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.hornet_block(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.0005)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = (data.to(self.device), target.to(self.device))
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()