# Auto-generated single-file for Atomref
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# Atomref
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding
from typing import Optional

# ---- Atomref (target) ----
class Atomref(torch.nn.Module):
    r"""Adds atom reference values to atomic energies.

    Args:
        atomref (torch.Tensor, optional):  A tensor of atom reference values,
            or :obj:`None` if not provided. (default: :obj:`None`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
    """
    def __init__(
        self,
        atomref: Optional[Tensor] = None,
        max_z: int = 100,
    ) -> None:
        super().__init__()

        if atomref is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = torch.as_tensor(atomref)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = Embedding(len(atomref), 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        r"""Adds atom reference values to atomic energies.

        Args:
            x (torch.Tensor): The atomic energies.
            z (torch.Tensor): The atomic numbers.
        """
        return x + self.atomref(z)


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self.atomref = Atomref()
        
        layers += [
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
