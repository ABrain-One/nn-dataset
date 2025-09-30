# Auto-generated single-file for RotaryEmbeddingMixed
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# h, w
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from typing import List
from typing import Union
from typing import Optional
def register_notrace_function(*args, **kwargs): pass
from typing import Tuple

# ---- timm.layers.pos_embed_sincos.rot ----
def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)

# ---- timm.layers.pos_embed_sincos.apply_rot_embed_cat ----
def apply_rot_embed_cat(x: torch.Tensor, emb):
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    return x * cos_emb + rot(x) * sin_emb

# ---- timm.layers.pos_embed_sincos.get_mixed_freqs ----
def get_mixed_freqs(
        freqs: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
) -> torch.Tensor:
    """Compute mixed (learnable) frequencies."""
    # Create position indices
    dtype = freqs.dtype
    freqs = freqs.float()
    freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
    freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
    combined = freqs_x + freqs_y  # shape: (num_heads, N, dim//4)
    sin_emb = torch.sin(combined).repeat_interleave(2, -1)  # (N, dim//2)
    cos_emb = torch.cos(combined).repeat_interleave(2, -1)  # (N, dim//2)
    rope_embeds = torch.cat([sin_emb, cos_emb], dim=-1)  # (num_heads, H*W, head_dim)
    return rope_embeds.to(dtype)

# ---- timm.layers.pos_embed_sincos.swap_shape_xy ----
def swap_shape_xy(seq: List[int]) -> List[int]:
    if len(seq) < 2:
        return seq
    return [seq[1], seq[0]] + list(seq[2:])

# ---- timm.layers.pos_embed_sincos.get_mixed_grid ----
def get_mixed_grid(
        shape: List[int],
        grid_indexing: str = 'ij',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if grid_indexing == 'xy':
        shape = swap_shape_xy(shape)
    x_pos, y_pos = torch.meshgrid(
        torch.arange(shape[0], dtype=dtype, device=device),
        torch.arange(shape[1], dtype=dtype, device=device),
        indexing=grid_indexing,
    )
    t_x = x_pos.flatten()
    t_y = y_pos.flatten()
    return t_x, t_y

# ---- timm.layers.pos_embed_sincos.init_random_2d_freqs ----
def init_random_2d_freqs(
        head_dim: int,
        depth: int,
        num_heads: int,
        temperature: float = 10.0,
        rotate: bool = True,
        *,
        device=None,
        dtype=torch.float32,
) -> torch.Tensor:
    """ Vectorised 2D ROPE frequencies with random rotation for mixed mode ROPE.
    Returns:
         Tensor (2, depth, num_heads, head_dim//2)
    """
    # base magnitudes, shape: (head_dim//4,)
    mag = 1.0 / (temperature ** (torch.arange(0, head_dim, 4, device=device, dtype=dtype) / head_dim))

    # (1,1,L) so it broadcasts over both depth and heads
    mag = mag.unsqueeze(0).unsqueeze(0)  # (1,1,L)

    # random (or zero) rotation per head *and* per block
    if rotate:
        angles = torch.rand(depth, num_heads, 1, device=device, dtype=dtype) * 2 * torch.pi
    else:
        angles = torch.zeros(depth, num_heads, 1, device=device, dtype=dtype)

    # build (depth, num_heads, 2·L) == head_dim//2 on the last axis
    fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(angles + torch.pi / 2)], dim=-1)
    fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(angles + torch.pi / 2)], dim=-1)

    # (2, depth, num_heads, head_dim//2)
    return torch.stack([fx, fy], dim=0)

# ---- RotaryEmbeddingMixed (target) ----
class RotaryEmbeddingMixed(nn.Module):
    """Rotary position embedding with depth-dependent learnable frequencies.

    This implementation supports mixed (learnable) ROPE. In mixed mode,
    each transformer block has its own set of learnable frequency parameters.

    Based on 'Rotary Position Embedding for Vision: https://arxiv.org/abs/2403.13298)'
    Compatible with original at https://github.com/naver-ai/rope-vit
    """
    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            temperature: float = 10.0,
            feat_shape: Optional[List[int]] = None,
            grid_indexing: str = 'xy',
    ):
        """Initialize rotary embeddings.

        Args:
            dim: Embedding dimension (should be divisible by 4)
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            temperature: Base for frequency computation
            feat_shape: Spatial dimensions [H, W] if known in advance
            grid_indexing: How to index grid positions ('xy' or 'ij')
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.temperature = temperature
        self.feat_shape = feat_shape
        self.grid_indexing = grid_indexing

        head_dim = dim // num_heads
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"

        freqs = init_random_2d_freqs(
            head_dim,
            depth,
            num_heads,
            temperature=temperature,
            rotate=True,
        )  # (2, depth, num_heads, head_dim//2)
        self.freqs = nn.Parameter(freqs)

        if feat_shape is not None:
            # cache pre-computed grid
            t_x, t_y = self._get_grid_values(feat_shape)
            self.register_buffer('t_x', t_x, persistent=False)
            self.register_buffer('t_y', t_y, persistent=False)
        else:
            self.t_x = self.t_y = None

    def _get_grid_values(self, feat_shape: Optional[List[int]]):
        t_x, t_y = get_mixed_grid(
            feat_shape,
            grid_indexing=self.grid_indexing,
            device=self.freqs.device
        )
        return t_x, t_y

    def update_feat_shape(self, feat_shape: Optional[List[int]]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            assert self.t_x is not None
            assert self.t_y is not None
            t_x, t_y = self._get_grid_values(feat_shape)
            self.t_x = t_x.to(self.t_x.device, self.t_x.dtype)
            self.t_y = t_y.to(self.t_y.device, self.t_y.dtype)
            self.feat_shape = feat_shape

    def get_embed(self, shape: Optional[List[int]] = None) -> torch.Tensor:
        """Generate rotary embeddings for the given spatial shape.

        Args:
            shape: Spatial dimensions [H, W]

        Returns:
            Tensor of shape (depth, H*W, dim) containing concatenated sin/cos embeddings
        """
        if shape is not None:
            t_x, t_y = get_mixed_grid(
                shape,
                grid_indexing=self.grid_indexing,
                device=self.freqs.device
            )
        elif self.t_x is not None and self.t_y is not None:
            t_x, t_y = self.t_x, self.t_y
        else:
            assert False, "get_embed() requires pre-computed t_x/t_y or valid shape"

        return get_mixed_freqs(self.freqs, t_x, t_y)

    def get_batch_embeds(
            self,
            shapes: List[Tuple[int, int]],
            seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate ROPE embeddings for multiple grid shapes efficiently.

        Computes embeddings for the maximum grid size once, then extracts
        and flattens the relevant portions for each requested shape.

        Args:
            shapes: List of (H, W) tuples representing different grid sizes
            seq_len: If provided, return padded tensor of this length. Otherwise return list.

        Returns:
            If seq_len is provided: Padded tensor of shape (len(shapes), depth, num_heads, seq_len, dim)
            Otherwise: List of tensors with shape (depth, num_heads, H*W, dim) for each shape
        """
        if not shapes:
            return []

        # Find max dimensions
        max_h = max(h for h, w in shapes)
        max_w = max(w for h, w in shapes)

        # Generate embeddings for max size ONCE
        t_x, t_y = get_mixed_grid(
            [max_h, max_w],
            grid_indexing=self.grid_indexing,
            device=self.freqs.device
        )
        max_embed = get_mixed_freqs(self.freqs, t_x, t_y)  # (depth, num_heads, max_h*max_w, dim)

        # Reshape to 2D grid for easy slicing
        depth, num_heads, _, dim = max_embed.shape
        max_embed_2d = max_embed.view(depth, num_heads, max_h, max_w, dim)

        if seq_len is not None:
            # Return padded tensor
            B = len(shapes)
            padded = torch.zeros(B, depth, num_heads, seq_len, dim, device=self.freqs.device, dtype=self.freqs.dtype)
            for i, (h, w) in enumerate(shapes):
                # Slice and flatten
                embed_slice = max_embed_2d[:, :, :h, :w].reshape(depth, num_heads, h * w, dim)
                actual_len = h * w
                padded[i, :, :, :actual_len] = embed_slice
            return padded
        else:
            # Return list
            results = []
            for h, w in shapes:
                # Slice and flatten
                embed_slice = max_embed_2d[:, :, :h, :w].reshape(depth, num_heads, h * w, dim)
                results.append(embed_slice)
            return results

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)

    def no_weight_decay(self):
        """Exclude frequency parameters from weight decay."""
        return {'freqs'}

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
        self.rotary_embedding_mixed = RotaryEmbeddingMixed(dim=64, depth=1, num_heads=1, temperature=10.0, feat_shape=[8, 8], grid_indexing='xy')
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
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        pos_embed = self.rotary_embedding_mixed.get_embed([H, W])
        x = apply_rot_embed_cat(x, pos_embed[0])  
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
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
