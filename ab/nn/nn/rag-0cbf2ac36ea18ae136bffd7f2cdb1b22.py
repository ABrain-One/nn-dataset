# Auto-generated single-file for Fuser
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
from torch import Tensor
import copy
import torch

# ---- original imports from contributing modules ----
from torch import Tensor, nn

# ---- Fuser (target) ----
class Fuser(nn.Module):
    """
    A module for fusing features through multiple layers of a neural network.

    This class applies a series of identical layers to an input tensor, optionally projecting the input first.

    Attributes:
        proj (nn.Module): An optional input projection layer. Identity if no projection is needed.
        layers (nn.ModuleList): A list of identical layers to be applied sequentially.

    Methods:
        forward: Applies the fuser to an input tensor.

    Examples:
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, layer: nn.Module, num_layers: int, dim: int | None = None, input_projection: bool = False):
        """
        Initialize the Fuser module for feature fusion through multiple layers.

        This module creates a sequence of identical layers and optionally applies an input projection.

        Args:
            layer (nn.Module): The layer to be replicated in the fuser.
            num_layers (int): The number of times to replicate the layer.
            dim (int | None): The dimension for input projection, if used.
            input_projection (bool): Whether to use input projection.

        Examples:
            >>> layer = nn.Linear(64, 64)
            >>> fuser = Fuser(layer, num_layers=3, dim=64, input_projection=True)
            >>> input_tensor = torch.randn(1, 64)
            >>> output = fuser(input_tensor)
        """
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a series of layers to the input tensor, optionally projecting it first."""
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
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
        layer = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.fuser = Fuser(layer, num_layers=2, dim=32, input_projection=True)
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
        x = self.fuser(x)
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
