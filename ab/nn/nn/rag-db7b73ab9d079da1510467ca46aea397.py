# Auto-generated single-file for CXBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm

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

# ---- ultralytics.models.sam.modules.blocks.DropPath ----
class DropPath(nn.Module):
    """
    Implements stochastic depth regularization for neural networks during training.

    Attributes:
        drop_prob (float): Probability of dropping a path during training.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    Methods:
        forward: Applies stochastic depth to input tensor during training, with optional scaling.

    Examples:
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initialize DropPath module for stochastic depth regularization during training."""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth to input tensor during training, with optional scaling."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

# ---- CXBlock (target) ----
class CXBlock(nn.Module):
    """
    ConvNeXt Block for efficient feature extraction in convolutional neural networks.

    This block implements a modified version of the ConvNeXt architecture, offering improved performance and
    flexibility in feature extraction.

    Attributes:
        dwconv (nn.Conv2d): Depthwise or standard 2D convolution layer.
        norm (LayerNorm2d): Layer normalization applied to channels.
        pwconv1 (nn.Linear): First pointwise convolution implemented as a linear layer.
        act (nn.GELU): GELU activation function.
        pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear layer.
        gamma (nn.Parameter | None): Learnable scale parameter for layer scaling.
        drop_path (nn.Module): DropPath layer for stochastic depth regularization.

    Methods:
        forward: Processes the input tensor through the ConvNeXt block.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        """
        Initialize a ConvNeXt Block for efficient feature extraction in convolutional neural networks.

        This block implements a modified version of the ConvNeXt architecture, offering improved performance and
        flexibility in feature extraction.

        Args:
            dim (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding size for the convolution.
            drop_path (float): Stochastic depth rate.
            layer_scale_init_value (float): Initial value for Layer Scale.
            use_dwconv (bool): Whether to use depthwise convolution.

        Examples:
            >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
            >>> x = torch.randn(1, 64, 32, 32)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 64, 32, 32])
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNeXt block operations to input tensor, including convolutions and residual connection."""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self.cx_block = CXBlock(dim=32, kernel_size=7, padding=3, drop_path=0.1)

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.cx_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
