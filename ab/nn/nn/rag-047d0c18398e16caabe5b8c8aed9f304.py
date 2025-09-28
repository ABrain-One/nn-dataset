# Auto-generated single-file for ImageEncoderViT
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

# ---- LayerNorm2d ----
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# ---- ultralytics.nn.modules.transformer.MLPBlock ----
class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """
        Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))

# ---- ultralytics.models.sam.modules.blocks.PatchEmbed ----
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding module for vision transformer architectures.

    This module converts an input image into a sequence of patch embeddings using a convolutional layer.
    It is commonly used as the first layer in vision transformer architectures to transform image data
    into a suitable format for subsequent transformer blocks.

    Attributes:
        proj (nn.Conv2d): Convolutional layer for projecting image patches to embeddings.

    Methods:
        forward: Applies patch embedding to the input tensor.

    Examples:
        >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
        torch.Size([1, 768, 14, 14])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Initialize the PatchEmbed module for converting image patches to embeddings.

        This module is typically used as the first layer in vision transformer architectures to transform
        image data into a suitable format for subsequent transformer blocks.

        Args:
            kernel_size (tuple[int, int]): Size of the convolutional kernel for patch extraction.
            stride (tuple[int, int]): Stride of the convolutional operation.
            padding (tuple[int, int]): Padding applied to the input before convolution.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimensionality of the output patch embeddings.

        Examples:
            >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = patch_embed(x)
            >>> print(output.shape)
            torch.Size([1, 768, 14, 14])
        """
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patch embedding by applying convolution and transposing resulting tensor."""
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C

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

# ---- ultralytics.models.sam.modules.blocks.REAttention ----
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

# ---- ultralytics.models.sam.modules.utils.window_partition ----
def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition input tensor into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input tensor with shape (B, H, W, C).
        window_size (int): Size of each window.

    Returns:
        windows (torch.Tensor): Partitioned windows with shape (B * num_windows, window_size, window_size, C).
        padded_h_w (tuple[int, int]): Padded height and width before partition.

    Examples:
        >>> x = torch.randn(1, 16, 16, 3)
        >>> windows, (Hp, Wp) = window_partition(x, window_size=4)
        >>> print(windows.shape, Hp, Wp)
        torch.Size([16, 4, 4, 3]) 16 16
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

# ---- ultralytics.models.sam.modules.utils.window_unpartition ----
def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]):
    """
    Unpartition windowed sequences into original sequences and remove padding.

    This function reverses the windowing process, reconstructing the original input from windowed segments
    and removing any padding that was added during the windowing process.

    Args:
        windows (torch.Tensor): Input tensor of windowed sequences with shape (B * num_windows, window_size,
            window_size, C), where B is the batch size, num_windows is the number of windows, window_size is
            the size of each window, and C is the number of channels.
        window_size (int): Size of each window.
        pad_hw (tuple[int, int]): Padded height and width (Hp, Wp) of the input before windowing.
        hw (tuple[int, int]): Original height and width (H, W) of the input before padding and windowing.

    Returns:
        (torch.Tensor): Unpartitioned sequences with shape (B, H, W, C), where B is the batch size, H and W
            are the original height and width, and C is the number of channels.

    Examples:
        >>> windows = torch.rand(32, 8, 8, 64)  # 32 windows of size 8x8 with 64 channels
        >>> pad_hw = (16, 16)  # Padded height and width
        >>> hw = (15, 14)  # Original height and width
        >>> x = window_unpartition(windows, window_size=8, pad_hw=pad_hw, hw=hw)
        >>> print(x.shape)
        torch.Size([1, 15, 14, 64])
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

# ---- ultralytics.models.sam.modules.blocks.Block ----
class Block(nn.Module):
    """
    Transformer block with support for window attention and residual propagation.

    This class implements a transformer block that can use either global or windowed self-attention,
    followed by a feed-forward network. It supports relative positional embeddings and is designed
    for use in vision transformer architectures.

    Attributes:
        norm1 (nn.Module): First normalization layer.
        attn (REAttention): Self-attention layer with optional relative positional encoding.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLPBlock): Multi-layer perceptron block.
        window_size (int): Size of attention window. If 0, global attention is used.

    Methods:
        forward: Processes input through the transformer block.

    Examples:
        >>> import torch
        >>> block = Block(dim=256, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Initialize a transformer block with optional window attention and relative positional embeddings.

        This constructor sets up a transformer block that can use either global or windowed self-attention,
        followed by a feed-forward network. It supports relative positional embeddings and is designed
        for use in vision transformer architectures.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in the self-attention layer.
            mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections.
            norm_layer (Type[nn.Module]): Type of normalization layer to use.
            act_layer (Type[nn.Module]): Type of activation function to use in the MLP block.
            use_rel_pos (bool): If True, uses relative positional embeddings in attention.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window. If 0, uses global attention.
            input_size (tuple[int, int] | None): Input resolution for calculating relative positional parameter size.

        Examples:
            >>> block = Block(dim=256, num_heads=8, window_size=7)
            >>> x = torch.randn(1, 56, 56, 256)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 56, 56, 256])
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = REAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through transformer block with optional windowed self-attention and residual connection."""
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))

# ---- ImageEncoderViT (target) ----
class ImageEncoderViT(nn.Module):
    """
    An image encoder using Vision Transformer (ViT) architecture for encoding images into a compact latent space.

    This class processes images by splitting them into patches, applying transformer blocks, and generating a final
    encoded representation through a neck module.

    Attributes:
        img_size (int): Dimension of input images, assumed to be square.
        patch_embed (PatchEmbed): Module for patch embedding.
        pos_embed (nn.Parameter | None): Absolute positional embedding for patches.
        blocks (nn.ModuleList): List of transformer blocks for processing patch embeddings.
        neck (nn.Sequential): Neck module to further process the output.

    Methods:
        forward: Process input through patch embedding, positional embedding, blocks, and neck.

    Examples:
        >>> import torch
        >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        >>> input_image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(input_image)
        >>> print(output.shape)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """
        Initialize an ImageEncoderViT instance for encoding images using Vision Transformer architecture.

        Args:
            img_size (int): Input image size, assumed to be square.
            patch_size (int): Size of image patches.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimension of patch embeddings.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads in each block.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            out_chans (int): Number of output channels from the neck module.
            qkv_bias (bool): If True, adds learnable bias to query, key, value projections.
            norm_layer (Type[nn.Module]): Type of normalization layer to use.
            act_layer (Type[nn.Module]): Type of activation layer to use.
            use_abs_pos (bool): If True, uses absolute positional embeddings.
            use_rel_pos (bool): If True, adds relative positional embeddings to attention maps.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window for windowed attention blocks.
            global_attn_indexes (tuple[int, ...]): Indices of blocks that use global attention.

        Examples:
            >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
            >>> input_image = torch.randn(1, 3, 224, 224)
            >>> output = encoder(input_image)
            >>> print(output.shape)
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through patch embedding, positional embedding, transformer blocks, and neck module."""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                F.interpolate(self.pos_embed.permute(0, 3, 1, 2), scale_factor=self.img_size / 1024).permute(0, 2, 3, 1)
                if self.img_size != 1024
                else self.pos_embed
            )
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))

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
        self.image_encoder_vit = ImageEncoderViT(img_size=32, patch_size=8, in_chans=self.in_channels, embed_dim=64, depth=1, num_heads=2, mlp_ratio=1.0, out_chans=32, qkv_bias=True, norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_abs_pos=False, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, global_attn_indexes=())
        self.classifier = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = self.image_encoder_vit(x)
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
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
