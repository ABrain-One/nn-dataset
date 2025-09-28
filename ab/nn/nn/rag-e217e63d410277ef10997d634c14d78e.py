# Auto-generated single-file for RotAttentionPool2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import os
import math
import warnings
import collections
from itertools import repeat
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

# ---- timm.layers.pos_embed_sincos.rot ----
def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)

# ---- timm.layers.pos_embed_sincos.apply_rot_embed ----
def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb

# ---- timm.layers.pos_embed_sincos.freq_bands ----
def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands

# ---- timm.layers.pos_embed_sincos.pixel_freq_bands ----
def pixel_freq_bands(
        num_bands: int,
        max_freq: float = 224.,
        linear_bands: bool = True,
        device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device)
    return bands * torch.pi

# ---- timm.layers.pos_embed_sincos.swap_shape_xy ----
def swap_shape_xy(seq: List[int]) -> List[int]:
    if len(seq) < 2:
        return seq
    return [seq[1], seq[0]] + list(seq[2:])

# ---- timm.layers.pos_embed_sincos.build_fourier_pos_embed ----
def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        num_bands: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                device=device,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
                device=device,
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if grid_indexing == 'xy':
        feat_shape = swap_shape_xy(feat_shape)
        if ref_feat_shape is not None:
            ref_feat_shape = swap_shape_xy(ref_feat_shape)

    if in_pixels:
        t = [
            torch.linspace(-1., 1., steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    else:
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) + grid_offset
            for s in feat_shape
        ]

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out

# ---- timm.layers.pos_embed_sincos.build_rotary_pos_embed ----
def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb

# ---- timm.layers.pos_embed_sincos.RotaryEmbedding ----
class RotaryEmbedding(nn.Module):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
            grid_offset: float = 0.,
            grid_indexing: str = 'ij',
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.linear_bands = linear_bands
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
            self.register_buffer(
                'bands',
                bands,
                persistent=False,
            )
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            emb_sin, emb_cos = self._get_pos_embed_values(feat_shape)
            self.bands = None
            self.register_buffer(
                'pos_embed_sin',
                emb_sin,
                persistent=False,
            )
            self.register_buffer(
                'pos_embed_cos',
                emb_cos,
                persistent=False,
            )

    def _get_pos_embed_values(self, feat_shape: List[int]):
        emb_sin, emb_cos = build_rotary_pos_embed(
            feat_shape=feat_shape,
            dim=self.dim,
            max_res=self.max_res,
            temperature=self.temperature,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
        )
        return emb_sin, emb_cos

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            # only update if feat_shape was set and different from previous value
            assert self.pos_embed_sin is not None
            assert self.pos_embed_cos is not None
            emb_sin, emb_cos = self._get_pos_embed_values(feat_shape)
            self.pos_embed_sin = emb_sin.to(self.pos_embed_sin.device, self.pos_embed_sin.dtype)
            self.pos_embed_cos = emb_cos.to(self.pos_embed_cos.device, self.pos_embed_cos.dtype)
            self.feat_shape = feat_shape

    def get_embed(self, shape: Optional[List[int]] = None):
        if shape is not None and self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                grid_offset=self.grid_offset,
                grid_indexing=self.grid_indexing,
            )
        elif self.pos_embed_sin is not None and self.pos_embed_cos is not None:
            return self.pos_embed_sin, self.pos_embed_cos
        else:
            assert False, "get_embed() requires pre-computed pos embeds or valid shape w/ pre-computed bands"

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)

# ---- timm.layers.weight_init._trunc_normal_ ----
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

# ---- timm.layers.weight_init.trunc_normal_ ----
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

# ---- timm.layers.config._EXPORTABLE ----
_EXPORTABLE = False

# ---- timm.layers.config._HAS_FUSED_ATTN ----
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

# ---- timm.layers.config._USE_FUSED_ATTN ----
_USE_FUSED_ATTN = int(os.environ.get('TIMM_FUSED_ATTN', '0'))

# ---- timm.layers.config.use_fused_attn ----
def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0

# ---- timm.layers.helpers.to_2tuple ----
to_2tuple = _ntuple(2)

# ---- RotAttentionPool2d (target) ----
class RotAttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: Optional[int] = None,
            ref_feat_size: Union[int, Tuple[int, int]] = 7,
            embed_dim: Optional[int] = None,
            head_dim: Optional[int] = 64,
            num_heads: Optional[int] = None,
            qkv_bias: bool = True,
            qkv_separate: bool = False,
            pool_type: str = 'token',
            class_token: bool = False,
            drop_rate: float = 0.,
    ):
        super().__init__()
        assert pool_type in ('', 'token')
        self.embed_dim = embed_dim = embed_dim or in_features
        self.in_features = in_features
        self.out_features = out_features or in_features
        ref_feat_size = to_2tuple(ref_feat_size)
        if num_heads is not None:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
        else:
            assert embed_dim % head_dim == 0
            num_heads = embed_dim // head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_type = pool_type.lower()
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        if class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        else:
            self.cls_token = None

        if qkv_separate:
            self.q = nn.Linear(in_features, embed_dim, bias=qkv_bias)
            self.k = nn.Linear(in_features, embed_dim, bias=qkv_bias)
            self.v = nn.Linear(in_features, embed_dim, bias=qkv_bias)
            self.qkv = None
        else:
            self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(embed_dim, self.out_features)
        self.pos_embed = RotaryEmbedding(self.head_dim, in_pixels=False, ref_feat_shape=ref_feat_size)

    def init_weights(self, zero_init_last: bool = False):
        if self.qkv is None:
            in_features = self.q.in_features
            trunc_normal_(self.q.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.q.bias)
            trunc_normal_(self.k.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.k.bias)
            trunc_normal_(self.v.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.v.bias)
        else:
            in_features = self.qkv.in_features
            trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
            nn.init.zeros_(self.qkv.bias)

    def reset(self, num_classes: Optional[int] = None, pool_type: Optional[str] = None):
        # NOTE: this module is being used as a head, so need compatible reset()
        if pool_type is not None:
            assert pool_type in ('', 'token')
            self.pool_type = pool_type
        if num_classes is not None:
            self.proj = nn.Linear(self.in_features, num_classes) if num_classes > 0 else nn.Identity()
            self.out_features = num_classes if num_classes > 0 else self.embed_dim

    def _pool(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if self.pool_type == 'token':
            x = x[:, 0]
        else:
            # if not pooled, return spatial output without token
            x = x[:, 1:].reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        return x

    def forward(self, x, pre_logits: bool = False):
        B, _, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is None:
            x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        else:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        if self.qkv is None:
            q = self.q(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = x.unbind(0)

        rse, rce = self.pos_embed.get_embed((H, W))
        q = torch.cat([q[:, :, :1, :], apply_rot_embed(q[:, :, 1:, :], rse, rce)], dim=2).type_as(v)
        k = torch.cat([k[:, :, :1, :], apply_rot_embed(k[:, :, 1:, :], rse, rce)], dim=2).type_as(v)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N + 1, -1)
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
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
        self.rot_attention_pool = RotAttentionPool2d(in_features=64, out_features=self.num_classes, ref_feat_size=4, embed_dim=32, head_dim=8, num_heads=4, qkv_bias=True, qkv_separate=False, pool_type='token', class_token=False, drop_rate=0.1)

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
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))  
        x = self.rot_attention_pool(x)
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
