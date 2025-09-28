# Auto-generated single-file for RelPosMlp
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import collections
from itertools import repeat
from functools import partial
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

# ---- timm.layers.grid.ndgrid ----
def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)

# ---- timm.layers.pos_embed_rel.gen_relative_log_coords ----
def gen_relative_log_coords(
        win_size: Tuple[int, int],
        pretrained_win_size: Tuple[int, int] = (0, 0),
        mode='swin',
):
    assert mode in ('swin', 'cr')
    # as per official swin-v2 impl, supporting timm specific 'cr' log coords as well
    relative_coords_h = torch.arange(-(win_size[0] - 1), win_size[0]).to(torch.float32)
    relative_coords_w = torch.arange(-(win_size[1] - 1), win_size[1]).to(torch.float32)
    relative_coords_table = torch.stack(ndgrid(relative_coords_h, relative_coords_w))
    relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()  # 2*Wh-1, 2*Ww-1, 2
    if mode == 'swin':
        if pretrained_win_size[0] > 0:
            relative_coords_table[:, :, 0] /= (pretrained_win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (pretrained_win_size[1] - 1)
        else:
            relative_coords_table[:, :, 0] /= (win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (win_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            1.0 + relative_coords_table.abs()) / math.log2(8)
    else:
        # mode == 'cr'
        relative_coords_table = torch.sign(relative_coords_table) * torch.log(
            1.0 + relative_coords_table.abs())

    return relative_coords_table

# ---- timm.layers.pos_embed_rel.gen_relative_position_index ----
def gen_relative_position_index(
        q_size: Tuple[int, int],
        k_size: Optional[Tuple[int, int]] = None,
        class_token: bool = False,
) -> torch.Tensor:
    # Adapted with significant modifications from Swin / BeiT codebases
    # get pair-wise relative position index for each token inside the window
    assert k_size is None, 'Different q & k sizes not currently supported'  # FIXME

    coords = torch.stack(ndgrid(torch.arange(q_size[0]), torch.arange(q_size[1]))).flatten(1)  # 2, Wh, Ww
    relative_coords = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0)  # Qh*Qw, Kh*Kw, 2
    relative_coords[:, :, 0] += q_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += q_size[1] - 1
    relative_coords[:, :, 0] *= 2 * q_size[1] - 1
    num_relative_distance = (2 * q_size[0] - 1) * (2 * q_size[1] - 1)

    # else:
    #     # FIXME different q vs k sizes is a WIP, need to better offset the two grids?
    #     q_coords = torch.stack(
    #         ndgrid(
    #             torch.arange(q_size[0]),
    #             torch.arange(q_size[1])
    #         )
    #     ).flatten(1)  # 2, Wh, Ww
    #     k_coords = torch.stack(
    #         ndgrid(
    #             torch.arange(k_size[0]),
    #             torch.arange(k_size[1])
    #         )
    #     ).flatten(1)
    #     relative_coords = q_coords[:, :, None] - k_coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    #     relative_coords = relative_coords.permute(1, 2, 0)  # Qh*Qw, Kh*Kw, 2
    #     relative_coords[:, :, 0] += max(q_size[0], k_size[0]) - 1  # shift to start from 0
    #     relative_coords[:, :, 1] += max(q_size[1], k_size[1]) - 1
    #     relative_coords[:, :, 0] *= k_size[1] + q_size[1] - 1
    #     relative_position_index = relative_coords.sum(-1)  # Qh*Qw, Kh*Kw
    #     num_relative_distance = (q_size[0] + k_size[0] - 1) * (q_size[1] + k_size[1] - 1) + 3

    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    if class_token:
        # handle cls to token & token 2 cls & cls to cls as per beit for rel pos bias
        # NOTE not intended or tested with MLP log-coords
        relative_position_index = F.pad(relative_position_index, [1, 0, 1, 0])
        relative_position_index[0, 0:] = num_relative_distance
        relative_position_index[0:, 0] = num_relative_distance + 1
        relative_position_index[0, 0] = num_relative_distance + 2

    return relative_position_index.contiguous()

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- timm.layers.mlp.Mlp ----
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# ---- RelPosMlp (target) ----
class RelPosMlp(nn.Module):
    """ Log-Coordinate Relative Position MLP
    Based on ideas presented in Swin-V2 paper (https://arxiv.org/abs/2111.09883)

    This impl covers the 'swin' implementation as well as two timm specific modes ('cr', and 'rw')
    """
    def __init__(
            self,
            window_size,
            num_heads=8,
            hidden_dim=128,
            prefix_tokens=0,
            mode='cr',
            pretrained_window_size=(0, 0)
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        if mode == 'swin':
            self.bias_act = nn.Sigmoid()
            self.bias_gain = 16
            mlp_bias = (True, False)
        else:
            self.bias_act = nn.Identity()
            self.bias_gain = None
            mlp_bias = True

        self.mlp = Mlp(
            2,  # x, y
            hidden_features=hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            bias=mlp_bias,
            drop=(0.125, 0.)
        )

        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(window_size).view(-1),
            persistent=False)

        # get relative_coords_table
        self.register_buffer(
            "rel_coords_log",
            gen_relative_log_coords(window_size, pretrained_window_size, mode=mode),
            persistent=False)

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log)
        if self.relative_position_index is not None:
            relative_position_bias = relative_position_bias.view(-1, self.num_heads)[self.relative_position_index]
            relative_position_bias = relative_position_bias.view(self.bias_shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = self.bias_act(relative_position_bias)
        if self.bias_gain is not None:
            relative_position_bias = self.bias_gain * relative_position_bias
        if self.prefix_tokens:
            relative_position_bias = F.pad(relative_position_bias, [self.prefix_tokens, 0, self.prefix_tokens, 0])
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()

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
        self.rel_pos_mlp = RelPosMlp(window_size=(4, 4), num_heads=8)
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
        x = F.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        attn = torch.randn(B, 8, H * W, H * W, device=x.device)
        attn = self.rel_pos_mlp(attn)
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
