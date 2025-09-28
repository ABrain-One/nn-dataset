# Auto-generated single-file for FpnNeck
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# bl, bx, by, nl, nx, ny
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# ---- ultralytics.models.sam.modules.blocks.PositionEmbeddingSine ----
class PositionEmbeddingSine(nn.Module):
    """
    A module for generating sinusoidal positional embeddings for 2D inputs like images.

    This class implements sinusoidal position encoding for 2D spatial positions, which can be used in
    transformer-based models for computer vision tasks.

    Attributes:
        num_pos_feats (int): Number of positional features (half of the embedding dimension).
        temperature (int): Temperature parameter for the sinusoidal functions.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): Scaling factor for the embeddings when normalize is True.
        cache (dict): Cache for storing precomputed embeddings.

    Methods:
        _encode_xy: Encodes 2D positions using sine and cosine functions.
        encode_boxes: Encodes box coordinates and dimensions into positional embeddings.
        encode_points: Encodes 2D point coordinates with sinusoidal positional embeddings.
        forward: Generates sinusoidal position embeddings for 2D inputs.

    Examples:
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 256, 224, 224])
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ):
        """Initialize sinusoidal position embeddings for 2D image inputs."""
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode 2D positions using sine/cosine functions for transformer positional embeddings."""
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Encode box coordinates and dimensions into positional embeddings for detection."""
        pos_x, pos_y = self._encode_xy(x, y)
        return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)

    encode = encode_boxes  # Backwards compatibility

    def encode_points(self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Encode 2D points with sinusoidal embeddings and append labels."""
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        return torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)

    def forward(self, x: torch.Tensor) -> Tensor:
        """Generate sinusoidal position embeddings for 2D inputs like images."""
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

# ---- FpnNeck (target) ----
class FpnNeck(nn.Module):
    """
    A Feature Pyramid Network (FPN) neck variant for multiscale feature fusion in object detection models.

    This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
    similar to ViT positional embedding interpolation.

    Attributes:
        position_encoding (PositionEmbeddingSine): Sinusoidal positional encoding module.
        convs (nn.ModuleList): List of convolutional layers for each backbone level.
        backbone_channel_list (list[int]): List of channel dimensions from the backbone.
        fpn_interp_model (str): Interpolation mode for FPN feature resizing.
        fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
        fpn_top_down_levels (list[int]): Levels to have top-down features in outputs.

    Methods:
        forward: Perform forward pass through the FPN neck.

    Examples:
        >>> backbone_channels = [64, 128, 256, 512]
        >>> fpn_neck = FpnNeck(256, backbone_channels)
        >>> inputs = [torch.rand(1, c, 32, 32) for c in backbone_channels]
        >>> outputs, positions = fpn_neck(inputs)
        >>> print(len(outputs), len(positions))
        4 4
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: list[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: list[int] | None = None,
    ):
        """
        Initialize a modified Feature Pyramid Network (FPN) neck.

        This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
        similar to ViT positional embedding interpolation.

        Args:
            d_model (int): Dimension of the model.
            backbone_channel_list (list[int]): List of channel dimensions from the backbone.
            kernel_size (int): Kernel size for the convolutional layers.
            stride (int): Stride for the convolutional layers.
            padding (int): Padding for the convolutional layers.
            fpn_interp_model (str): Interpolation mode for FPN feature resizing.
            fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
            fpn_top_down_levels (Optional[list[int]]): Levels to have top-down features in outputs.

        Examples:
            >>> backbone_channels = [64, 128, 256, 512]
            >>> fpn_neck = FpnNeck(256, backbone_channels)
            >>> print(fpn_neck)
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {"sum", "avg"}
        self.fuse_type = fuse_type

        # Levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level
        if fpn_top_down_levels is None:
            # Default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: list[torch.Tensor]):
        """
        Perform forward pass through the Feature Pyramid Network (FPN) neck.

        This method processes a list of input tensors from the backbone through the FPN, applying lateral connections
        and top-down feature fusion. It generates output feature maps and corresponding positional encodings.

        Args:
            xs (list[torch.Tensor]): List of input tensors from the backbone, each with shape (B, C, H, W).

        Returns:
            out (list[torch.Tensor]): List of output feature maps after FPN processing, each with shape
                (B, d_model, H, W).
            pos (list[torch.Tensor]): List of positional encodings corresponding to each output feature map.

        Examples:
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
            >>> print(len(outputs), len(positions))
            4 4
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # FPN forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # Forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=x.dtype),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos

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
        self.fpn_neck = FpnNeck(d_model=32, backbone_channel_list=[32, 32, 32, 32])
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # Create multi-scale features by downsampling
        x1 = x  # 32x32
        x2 = torch.nn.functional.avg_pool2d(x, 2)  # 16x16
        x3 = torch.nn.functional.avg_pool2d(x, 4)  # 8x8
        x4 = torch.nn.functional.avg_pool2d(x, 8)  # 4x4
        xs = [x1, x2, x3, x4]
        outputs, positions = self.fpn_neck(xs)
        x = outputs[0]
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
