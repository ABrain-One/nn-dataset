# Auto-generated single-file for KLDivLoss
# Dependencies are emitted in topological order (utilities first).

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

# ---- KLDivLoss (target) ----
class KLDivLoss(_Loss):
    r"""The Kullback-Leibler divergence loss.

    For tensors of the same shape :math:`y_{\text{pred}},\ y_{\text{true}}`,
    where :math:`y_{\text{pred}}` is the :attr:`input` and :math:`y_{\text{true}}` is the
    :attr:`target`, we define the **pointwise KL-divergence** as

    .. math::

        L(y_{\text{pred}},\ y_{\text{true}})
            = y_{\text{true}} \cdot \log \frac{y_{\text{true}}}{y_{\text{pred}}}
            = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})

    To avoid underflow issues when computing this quantity, this loss expects the argument
    :attr:`input` in the log-space. The argument :attr:`target` may also be provided in the
    log-space if :attr:`log_target`\ `= True`.

    To summarise, this function is roughly equivalent to computing

    .. code-block:: python

        if not log_target:  # default
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

    and then reducing this result depending on the argument :attr:`reduction` as

    .. code-block:: python

        if reduction == "mean":  # default
            loss = loss_pointwise.mean()
        elif reduction == "batchmean":  # mathematically correct
            loss = loss_pointwise.sum() / input.size(0)
        elif reduction == "sum":
            loss = loss_pointwise.sum()
        else:  # reduction == "none"
            loss = loss_pointwise

    .. note::
        As all the other losses in PyTorch, this function expects the first argument,
        :attr:`input`, to be the output of the model (e.g. the neural network)
        and the second, :attr:`target`, to be the observations in the dataset.
        This differs from the standard mathematical notation :math:`KL(P\ ||\ Q)` where
        :math:`P` denotes the distribution of the observations and :math:`Q` denotes the model.

    .. warning::
        :attr:`reduction`\ `= "mean"` doesn't return the true KL divergence value, please use
        :attr:`reduction`\ `= "batchmean"` which aligns with the mathematical definition.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to `False`, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is `False`. Default: `True`
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is `False`, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: `True`
        reduction (str, optional): Specifies the reduction to apply to the output. Default: `"mean"`
        log_target (bool, optional): Specifies whether `target` is the log space. Default: `False`

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar by default. If :attr:`reduction` is `'none'`, then :math:`(*)`,
          same shape as the input.

    Examples:
        >>> kl_loss = nn.KLDivLoss(reduction="batchmean")
        >>> # input should be a distribution in the log space
        >>> input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
        >>> # Sample a batch of distributions. Usually this would come from the dataset
        >>> target = F.softmax(torch.rand(3, 5), dim=1)
        >>> output = kl_loss(input, target)
        >>>
        >>> kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        >>> log_target = F.log_softmax(torch.rand(3, 5), dim=1)
        >>> output = kl_loss(input, log_target)
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.kl_div(
            input, target, reduction=self.reduction, log_target=self.log_target
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
        self.kl_div_loss = KLDivLoss(reduction='mean', log_target=False)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        log_pred = torch.randn(x.size(0), self.num_classes, device=x.device)
        target_dist = torch.randn(x.size(0), self.num_classes, device=x.device)
        target_dist = nn.functional.softmax(target_dist, dim=1)
        
        kl_loss = self.kl_div_loss(log_pred, target_dist)
        x_combined = x + kl_loss.unsqueeze(-1)
        return self.classifier(x_combined)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
