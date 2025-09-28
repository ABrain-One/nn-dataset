# Auto-generated single-file for ExpNormalSmearing
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Tuple
import math

# ---- torch_geometric.nn.models.visnet.CosineCutoff ----
class CosineCutoff(torch.nn.Module):
    r"""Applies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point at which the cutoff
            is applied.
    """
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances.

        Returns:
            cutoffs (torch.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

# ---- ExpNormalSmearing (target) ----
class ExpNormalSmearing(torch.nn.Module):
    r"""Applies exponential normal smearing to the input distances.

    .. math::
        \text{smeared\_dist} = \text{CosineCutoff}(\text{dist})
        * e^{-\beta * (e^{\alpha * (-\text{dist})} - \text{means})^2}

    Args:
        cutoff (float, optional): A scalar that determines the point at which
            the cutoff is applied. (default: :obj:`5.0`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`128`)
        trainable (bool, optional): If set to :obj:`False`, the means and betas
            of the RBFs will not be trained. (default: :obj:`True`)
    """
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter('means', Parameter(means))
            self.register_parameter('betas', Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        r"""Initializes the means and betas for the radial basis functions."""
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        r"""Resets the means and betas to their initial values."""
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        r"""Applies the exponential normal smearing to the input distance.

        Args:
            dist (torch.Tensor): A tensor of distances.
        """
        dist = dist.unsqueeze(-1)
        smeared_dist = self.cutoff_fn(dist) * (-self.betas * (
            (self.alpha * (-dist)).exp() - self.means)**2).exp()
        return smeared_dist

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
        self.exp_normal_smearing = ExpNormalSmearing(cutoff=5.0, num_rbf=32, trainable=True)
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        distances = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1)
        smeared_dist = self.exp_normal_smearing(distances)
        x = smeared_dist.mean(dim=1)
        
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
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
