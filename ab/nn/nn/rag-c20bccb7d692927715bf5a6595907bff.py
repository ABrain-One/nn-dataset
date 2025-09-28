# Auto-generated single-file for RotaryEmbedding
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import List, Optional
import math
from typing import List
from typing import Optional

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

# ---- RotaryEmbedding (target) ----
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
        self.rotary_embedding = RotaryEmbedding(dim=64, max_res=224, temperature=10000, in_pixels=True, linear_bands=False, feat_shape=[8, 8], ref_feat_shape=None, grid_offset=0.0, grid_indexing='ij')
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
        x = self.rotary_embedding(x)
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
