# Auto-generated single-file for SELU
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
import torch

# ---- original imports from contributing modules ----

# ---- SELU (target) ----
class SELU(Module):
    r"""Applies the SELU function element-wise.

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    .. warning::
        When using ``kaiming_normal`` or ``kaiming_normal_`` for initialisation,
        ``nonlinearity='linear'`` should be used instead of ``nonlinearity='selu'``
        in order to get `Self-Normalizing Neural Networks`_.
        See :func:`torch.nn.init.calculate_gain` for more information.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SELU.png

    Examples::

        >>> m = nn.SELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.selu = SELU(inplace=True)
        self.features = self.build_features()
        import torch.nn as nn
        self.classifier_weight = nn.Parameter(torch.randn(self.num_classes, 64))
        self.classifier_bias = nn.Parameter(torch.randn(self.num_classes))

    def build_features(self):
        import torch.nn as nn
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            self.selu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.selu,
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        x = F.linear(x, self.classifier_weight, self.classifier_bias)
        return x

    def train_setup(self, prm: dict):
        import torch.optim as optim
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = F.cross_entropy

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
