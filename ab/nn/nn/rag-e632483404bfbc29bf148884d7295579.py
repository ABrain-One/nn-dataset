# Auto-generated single-file for Sphere
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor
import math

# ---- original imports from contributing modules ----

# ---- Sphere (target) ----
class Sphere(torch.nn.Module):
    r"""Computes spherical harmonics of the input data.

    This module computes the spherical harmonics up to a given degree
    :obj:`lmax` for the input tensor of 3D vectors.
    The vectors are assumed to be given in Cartesian coordinates.
    See `here <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>`_
    for mathematical details.

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`2`)
    """
    def __init__(self, lmax: int = 2) -> None:
        super().__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        r"""Computes the spherical harmonics of the input tensor.

        Args:
            edge_vec (torch.Tensor): A tensor of 3D vectors.
        """
        return self._spherical_harmonics(
            self.lmax,
            edge_vec[..., 0],
            edge_vec[..., 1],
            edge_vec[..., 2],
        )

    def _spherical_harmonics(
        self,
        lmax: int,
        x: Tensor,
        y: Tensor,
        z: Tensor,
    ) -> Tensor:
        r"""Computes the spherical harmonics up to degree :obj:`lmax` of the
        input vectors.

        Args:
            lmax (int): The maximum degree of the spherical harmonics.
            x (torch.Tensor): The x coordinates of the vectors.
            y (torch.Tensor): The y coordinates of the vectors.
            z (torch.Tensor): The z coordinates of the vectors.
        """
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
            ], dim=-1)

        raise ValueError(f"'lmax' needs to be 1 or 2 (got {lmax})")

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.sphere = Sphere(lmax=2)
        self.classifier = torch.nn.Linear(8, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_3d = torch.cat([x_flat, torch.zeros(B, H * W, 1, device=x.device)], dim=-1)
        x_sphere = self.sphere(x_3d)
        x = x_sphere.mean(dim=1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
