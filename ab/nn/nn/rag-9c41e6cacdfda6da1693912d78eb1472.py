# Auto-generated single-file for HuberLoss
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# _Reduction
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import _Reduction

# ---- torch.nn.modules.loss._Loss ----
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

# ---- HuberLoss (target) ----
class HuberLoss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.
    This loss combines advantages of both :class:`L1Loss` and :class:`MSELoss`; the
    delta-scaled L1 region makes the loss less sensitive to outliers than :class:`MSELoss`,
    while the L2 region provides smoothness over :class:`L1Loss` near 0. See
    `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^T

    with

    .. math::
        l_n = \begin{cases}
        0.5 (x_n - y_n)^2, & \text{if } |x_n - y_n| < delta \\
        delta * (|x_n - y_n| - 0.5 * delta), & \text{otherwise }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    .. note::
        When delta is set to 1, this loss is equivalent to :class:`SmoothL1Loss`.
        In general, this loss differs from :class:`SmoothL1Loss` by a factor of delta (AKA beta
        in Smooth L1).
        See :class:`SmoothL1Loss` for additional discussion on the differences in behavior
        between the two losses.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
            The value must be positive.  Default: 1.0

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same shape as the input.
    """

    __constants__ = ["reduction", "delta"]

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)

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
        self.huber_loss = HuberLoss(reduction='mean', delta=1.0)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
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
