# Auto-generated single-file for TinyViTBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import *

# ---- ultralytics.models.sam.modules.tiny_encoder.Attention ----
class Attention(torch.nn.Module):
    """
    Multi-head attention module with spatial awareness and trainable attention biases.

    This module implements a multi-head attention mechanism with support for spatial awareness, applying
    attention biases based on spatial resolution. It includes trainable attention biases for each unique
    offset between spatial positions in the resolution grid.

    Attributes:
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for attention scores.
        key_dim (int): Dimensionality of the keys and queries.
        nh_kd (int): Product of num_heads and key_dim.
        d (int): Dimensionality of the value vectors.
        dh (int): Product of d and num_heads.
        attn_ratio (float): Attention ratio affecting the dimensions of the value vectors.
        norm (nn.LayerNorm): Layer normalization applied to input.
        qkv (nn.Linear): Linear layer for computing query, key, and value projections.
        proj (nn.Linear): Linear layer for final projection.
        attention_biases (nn.Parameter): Learnable attention biases.
        attention_bias_idxs (torch.Tensor): Indices for attention biases.
        ab (torch.Tensor): Cached attention biases for inference, deleted during training.

    Examples:
        >>> attn = Attention(dim=256, key_dim=64, num_heads=8, resolution=(14, 14))
        >>> x = torch.randn(1, 196, 256)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 196, 256])
    """

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4,
        resolution: tuple[int, int] = (14, 14),
    ):
        """
        Initialize the Attention module for multi-head attention with spatial awareness.

        This module implements a multi-head attention mechanism with support for spatial awareness, applying
        attention biases based on spatial resolution. It includes trainable attention biases for each unique
        offset between spatial positions in the resolution grid.

        Args:
            dim (int): The dimensionality of the input and output.
            key_dim (int): The dimensionality of the keys and queries.
            num_heads (int, optional): Number of attention heads.
            attn_ratio (float, optional): Attention ratio, affecting the dimensions of the value vectors.
            resolution (tuple[int, int], optional): Spatial resolution of the input feature map.
        """
        super().__init__()

        assert isinstance(resolution, tuple) and len(resolution) == 2, "'resolution' argument not tuple of length 2"
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)

    def train(self, mode: bool = True):
        """Set the module in training mode and handle the 'ab' attribute for cached attention biases."""
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention with spatial awareness and trainable attention biases."""
        B, N, _ = x.shape  # B, N, C

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        self.ab = self.ab.to(self.attention_biases.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        return self.proj(x)

# ---- ultralytics.models.sam.modules.tiny_encoder.Conv2d_BN ----
class Conv2d_BN(torch.nn.Sequential):
    """
    A sequential container that performs 2D convolution followed by batch normalization.

    This module combines a 2D convolution layer with batch normalization, providing a common building block
    for convolutional neural networks. The batch normalization weights and biases are initialized to specific
    values for optimal training performance.

    Attributes:
        c (torch.nn.Conv2d): 2D convolution layer.
        bn (torch.nn.BatchNorm2d): Batch normalization layer.

    Examples:
        >>> conv_bn = Conv2d_BN(3, 64, ks=3, stride=1, pad=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = conv_bn(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 64, 224, 224])
    """

    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1,
    ):
        """
        Initialize a sequential container with 2D convolution followed by batch normalization.

        Args:
            a (int): Number of input channels.
            b (int): Number of output channels.
            ks (int, optional): Kernel size for the convolution.
            stride (int, optional): Stride for the convolution.
            pad (int, optional): Padding for the convolution.
            dilation (int, optional): Dilation factor for the convolution.
            groups (int, optional): Number of groups for the convolution.
            bn_weight_init (float, optional): Initial value for batch normalization weight.
        """
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

# ---- ultralytics.models.sam.modules.tiny_encoder.MLP ----
class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) module for transformer architectures.

    This module applies layer normalization, two fully-connected layers with an activation function in between,
    and dropout. It is commonly used in transformer-based architectures for processing token embeddings.

    Attributes:
        norm (nn.LayerNorm): Layer normalization applied to the input.
        fc1 (nn.Linear): First fully-connected layer.
        fc2 (nn.Linear): Second fully-connected layer.
        act (nn.Module): Activation function applied after the first fully-connected layer.
        drop (nn.Dropout): Dropout layer applied after the activation function.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> mlp = MLP(in_features=256, hidden_features=512, out_features=256, activation=nn.GELU, drop=0.1)
        >>> x = torch.randn(32, 100, 256)
        >>> output = mlp(x)
        >>> print(output.shape)
        torch.Size([32, 100, 256])
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation=nn.GELU,
        drop: float = 0.0,
    ):
        """
        Initialize a multi-layer perceptron with configurable input, hidden, and output dimensions.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int], optional): Number of hidden features.
            out_features (Optional[int], optional): Number of output features.
            activation (nn.Module): Activation function applied after the first fully-connected layer.
            drop (float, optional): Dropout probability.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = activation()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP operations: layer norm, FC layers, activation, and dropout to the input tensor."""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

# ---- TinyViTBlock (target) ----
class TinyViTBlock(nn.Module):
    """
    TinyViT Block that applies self-attention and a local convolution to the input.

    This block is a key component of the TinyViT architecture, combining self-attention mechanisms with
    local convolutions to process input features efficiently. It supports windowed attention for
    computational efficiency and includes residual connections.

    Attributes:
        dim (int): The dimensionality of the input and output.
        input_resolution (tuple[int, int]): Spatial resolution of the input feature map.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop_path (nn.Module): Stochastic depth layer, identity function during inference.
        attn (Attention): Self-attention module.
        mlp (MLP): Multi-layer perceptron module.
        local_conv (Conv2d_BN): Depth-wise local convolution layer.

    Examples:
        >>> input_tensor = torch.randn(1, 196, 192)
        >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3)
        >>> output = block(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 196, 192])
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation=nn.GELU,
    ):
        """
        Initialize a TinyViT block with self-attention and local convolution.

        This block is a key component of the TinyViT architecture, combining self-attention mechanisms with
        local convolutions to process input features efficiently.

        Args:
            dim (int): Dimensionality of the input and output features.
            input_resolution (tuple[int, int]): Spatial resolution of the input feature map (height, width).
            num_heads (int): Number of attention heads.
            window_size (int, optional): Size of the attention window. Must be greater than 0.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension.
            drop (float, optional): Dropout rate.
            drop_path (float, optional): Stochastic depth rate.
            local_conv_size (int, optional): Kernel size of the local convolution.
            activation (nn.Module): Activation function for MLP.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # NOTE: `DropPath` is needed only for training.
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, activation=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention, local convolution, and MLP operations to the input tensor."""
        h, w = self.input_resolution
        b, hw, c = x.shape  # batch, height*width, channels
        assert hw == h * w, "input feature has wrong size"
        res_x = x
        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = h + pad_b, w + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # Window partition
            x = (
                x.view(b, nH, self.window_size, nW, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nH * nW, self.window_size * self.window_size, c)
            )
            x = self.attn(x)

            # Window reverse
            x = x.view(b, nH, nW, self.window_size, self.window_size, c).transpose(2, 3).reshape(b, pH, pW, c)
            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, hw, c)

        x = res_x + self.drop_path(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, hw).transpose(1, 2)

        return x + self.drop_path(self.mlp(x))

    def extra_repr(self) -> str:
        """
        Return a string representation of the TinyViTBlock's parameters.

        This method provides a formatted string containing key information about the TinyViTBlock, including its
        dimension, input resolution, number of attention heads, window size, and MLP ratio.

        Returns:
            (str): A formatted string containing the block's parameters.

        Examples:
            >>> block = TinyViTBlock(dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0)
            >>> print(block.extra_repr())
            dim=192, input_resolution=(14, 14), num_heads=3, window_size=7, mlp_ratio=4.0
        """
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )

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
        
        self.tiny_vit_block = TinyViTBlock(
            dim=64,
            input_resolution=(8, 8),
            num_heads=4,
            window_size=4,
            mlp_ratio=2.0,
            drop=0.1,
            drop_path=0.1,
            local_conv_size=3,
            activation=nn.GELU
        )
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
        x = F.adaptive_avg_pool2d(x, (8, 8))
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        
        x = self.tiny_vit_block(x)
        x = x.mean(dim=1)
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
