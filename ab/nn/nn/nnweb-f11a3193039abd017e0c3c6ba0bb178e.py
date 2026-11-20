import math
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]
    x1 = x[..., 1::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, :]
    x3 = x[..., 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], -1)
    return x


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
        relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMergingV2(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: Tensor):
        x = _patch_merging_pad(x)
        x = self.reduction(x)
        x = self.norm(x)
        return x


def shifted_window_attention(
        input: Tensor,
        qkv_weight: Tensor,
        proj_weight: Tensor,
        relative_position_bias: Tensor,
        window_size: List[int],
        num_heads: int,
        shift_size: List[int],
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        qkv_bias: Optional[Tensor] = None,
        proj_bias: Optional[Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        training: bool = True,
) -> Tensor:
    B, H, W, C = input.shape
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length: 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0]: h[1], w[0]: w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            window_size: List[int],
            shift_size: List[int],
            num_heads: int,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            attention_dropout: float = 0.0,
            dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size
        )

    def forward(self, x: Tensor) -> Tensor:
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    def __init__(
            self,
            dim: int,
            window_size: List[int],
            shift_size: List[int],
            num_heads: int,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            attention_dropout: float = 0.0,
            dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length: 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8
        relative_coords_table = (
                torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor):
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class SwinTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: List[int],
            shift_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.0,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockV2(SwinTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: List[int],
            shift_size: List[int],
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.0,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        
        # Hyperparameters
        dropout: float = prm['dropout']
        attention_dropout: float = prm['attention_dropout']
        stochastic_depth_prob: float = prm['stochastic_depth_prob']
        num_classes: int = out_shape[0]
        in_channels: int = in_shape[1]

        # Architecture: 4 Conv stages, up to 256 channels
        # Stage 1: 32 channels
        # Stage 2: 64 channels
        # Stage 3: 128 channels
        # Stage 4: 256 channels
        
        # Standard choices for "residual-style skip connection in deeper feature stages"
        # and "batch normalization after convolution blocks"
        
        # Helper to create a residual block
        def make_res_block(in_ch, out_ch, stride=1):
            # If stride is not 1 or channels change, we need a projection for the skip connection
            # Standard ResNet style: 1x1 conv for projection
            shortcut = nn.Identity()
            if stride != 1 or in_ch != out_ch:
                shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
            
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout), # Using dropout as requested
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
            # Residual connection
            # We need to handle the addition. 
            # Since we are building a Sequential-like structure manually or using a custom module, 
            # let's create a simple module for the block.
            
            class ResBlock(nn.Module):
                def __init__(self, block, shortcut):
                    super().__init__()
                    self.block = block
                    self.shortcut = shortcut
                    self.relu = nn.ReLU(inplace=True)
                
                def forward(self, x):
                    identity = self.shortcut(x)
                    out = self.block(x)
                    out = out + identity
                    out = self.relu(out)
                    return out
            
            return ResBlock(block, shortcut)

        # Stage 1: 3 -> 32, stride 1
        # Description says "residual-style skip connection in deeper feature stages". 
        # Usually stage 1 is just conv+bn+relu. Let's start with a plain conv for stage 1.
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

        # Stage 2: 32 -> 64, stride 2
        # Deeper stage, so use residual blocks.
        # Two blocks of 64 channels.
        self.stage2 = nn.Sequential(
            make_res_block(32, 64, stride=2),
            make_res_block(64, 64, stride=1)
        )

        # Stage 3: 64 -> 128, stride 2
        # Two blocks of 128 channels.
        self.stage3 = nn.Sequential(
            make_res_block(64, 128, stride=2),
            make_res_block(128, 128, stride=1)
        )

        # Stage 4: 128 -> 256, stride 2
        # Two blocks of 256 channels.
        self.stage4 = nn.Sequential(
            make_res_block(128, 256, stride=2),
            make_res_block(256, 256, stride=1)
        )

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(256, num_classes)

        # We must use 'attention_dropout' and 'stochastic_depth_prob' even though they are transformer-specific.
        # Since this is a CNN, we can't really use attention dropout in the same way.
        # However, the rule says "each of those names ... must still be read from prm and actually used in the new model".
        # We can apply a dropout with p=attention_dropout to the features, or just store it. 
        # "Actually used" implies it affects the forward pass. 
        # Let's add a dropout layer with p=attention_dropout after the final stage.
        self.attn_dropout = nn.Dropout(p=attention_dropout)
        
        # Stochastic depth is hard to apply to CNNs in the same "row" manner. 
        # We can approximate by applying a dropout that mimics stochastic depth (dropping entire samples) 
        # or just using a standard dropout. Given the constraint, let's apply a dropout 
        # with p=stochastic_depth_prob to the output of the stages or before the head.
        # To be safe and "actually use" it, let's add a dropout layer with p=stochastic_depth_prob.
        self.stoch_dropout = nn.Dropout(p=stochastic_depth_prob)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Use the hyperparameters that don't fit naturally into the CNN blocks
        x = self.attn_dropout(x)
        x = self.stoch_dropout(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x