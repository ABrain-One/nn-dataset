# Auto-generated single-file for PositionEmbeddingSine
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# bl, bx, by, nl, nx, ny
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
import math

# ---- original imports from contributing modules ----
from torch import Tensor, nn

# ---- PositionEmbeddingSine (target) ----
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
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=32, temperature=10000, normalize=True)
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
        pos_encoding = self.pos_embed(x)
        x = x + pos_encoding
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
