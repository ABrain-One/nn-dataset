# Auto-generated single-file for REAttention
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- ultralytics.models.sam.modules.utils.get_rel_pos ----
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Extract relative positional embeddings based on query and key sizes.

    Args:
        q_size (int): Size of the query.
        k_size (int): Size of the key.
        rel_pos (torch.Tensor): Relative position embeddings with shape (L, C), where L is the maximum relative
            distance and C is the embedding dimension.

    Returns:
        (torch.Tensor): Extracted positional embeddings according to relative positions, with shape (q_size,
            k_size, C).

    Examples:
        >>> q_size, k_size = 8, 16
        >>> rel_pos = torch.randn(31, 64)  # 31 = 2 * max(8, 16) - 1
        >>> extracted_pos = get_rel_pos(q_size, k_size, rel_pos)
        >>> print(extracted_pos.shape)
        torch.Size([8, 16, 64])
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

# ---- ultralytics.models.sam.modules.utils.add_decomposed_rel_pos ----
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """
    Add decomposed Relative Positional Embeddings to the attention map.

    This function calculates and applies decomposed Relative Positional Embeddings as described in the MVITv2
    paper. It enhances the attention mechanism by incorporating spatial relationships between query and key
    positions.

    Args:
        attn (torch.Tensor): Attention map with shape (B, q_h * q_w, k_h * k_w).
        q (torch.Tensor): Query tensor in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings for height axis with shape (Lh, C).
        rel_pos_w (torch.Tensor): Relative position embeddings for width axis with shape (Lw, C).
        q_size (tuple[int, int]): Spatial sequence size of query q as (q_h, q_w).
        k_size (tuple[int, int]): Spatial sequence size of key k as (k_h, k_w).

    Returns:
        (torch.Tensor): Updated attention map with added relative positional embeddings, shape
            (B, q_h * q_w, k_h * k_w).

    Examples:
        >>> B, C, q_h, q_w, k_h, k_w = 1, 64, 8, 8, 8, 8
        >>> attn = torch.rand(B, q_h * q_w, k_h * k_w)
        >>> q = torch.rand(B, q_h * q_w, C)
        >>> rel_pos_h = torch.rand(2 * max(q_h, k_h) - 1, C)
        >>> rel_pos_w = torch.rand(2 * max(q_w, k_w) - 1, C)
        >>> q_size, k_size = (q_h, q_w), (k_h, k_w)
        >>> updated_attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)
        >>> print(updated_attn.shape)
        torch.Size([1, 64, 64])

    References:
        https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn

# ---- REAttention (target) ----
class REAttention(nn.Module):
    """
    Relative Position Attention module for efficient self-attention in transformer architectures.

    This class implements a multi-head attention mechanism with relative positional embeddings, designed
    for use in vision transformer models. It supports optional query pooling and window partitioning
    for efficient processing of large inputs.

    Attributes:
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for attention computation.
        qkv (nn.Linear): Linear projection for query, key, and value.
        proj (nn.Linear): Output projection layer.
        use_rel_pos (bool): Whether to use relative positional embeddings.
        rel_pos_h (nn.Parameter): Relative positional embeddings for height dimension.
        rel_pos_w (nn.Parameter): Relative positional embeddings for width dimension.

    Methods:
        forward: Applies multi-head attention with optional relative positional encoding to input tensor.

    Examples:
        >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
        >>> x = torch.randn(1, 32, 32, 256)
        >>> output = attention(x)
        >>> print(output.shape)
        torch.Size([1, 32, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Initialize a Relative Position Attention module for transformer-based architectures.

        This module implements multi-head attention with optional relative positional encodings, designed
        specifically for vision tasks in transformer models.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections.
            use_rel_pos (bool): If True, uses relative positional encodings.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            input_size (tuple[int, int] | None): Input resolution for calculating relative positional parameter size.
                Required if use_rel_pos is True.

        Examples:
            >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
            >>> x = torch.randn(1, 32, 32, 256)
            >>> output = attention(x)
            >>> print(output.shape)
            torch.Size([1, 32, 32, 256])
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # Initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention with optional relative positional encoding to input tensor."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)

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
        self.re_attention = REAttention(dim=32, num_heads=1, qkv_bias=True, use_rel_pos=False, rel_pos_zero_init=True, input_size=(4, 4))
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = F.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = self.re_attention(x)
        
        x = x.permute(0, 3, 1, 2)
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
