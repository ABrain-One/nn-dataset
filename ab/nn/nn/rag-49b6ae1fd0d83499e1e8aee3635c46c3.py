# Auto-generated single-file for CrossEntropyLoss
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from typing import Optional

# ---- _Reduction ----
class _Reduction:
    @staticmethod
    def legacy_get_string(size_average, reduce):
        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True
        if size_average and reduce:
            return 'mean'
        elif reduce:
            return 'sum'
        else:
            return 'none'

# ---- torch.nn.modules.loss._Loss ----
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

# ---- torch.nn.modules.loss._WeightedLoss ----
class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Optional[Tensor]

# ---- CrossEntropyLoss (target) ----
class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion computes the cross entropy loss between input logits
    and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain the unnormalized logits for each class (which do `not` need
    to be positive or sum to 1, in general).
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    `K`-dimensional case. The last being useful for higher dimension inputs, such
    as computing cross entropy loss per-pixel for 2D images.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to applying :class:`~torch.nn.LogSoftmax`
      on an input, followed by :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`. The
          target data type is required to be long when using class indices. If containing class probabilities, the
          target must be the same shape input, and each value should be between :math:`[0, 1]`. This means the target
          data type is required to be float when using class probabilities. Note that PyTorch does not strictly enforce
          probability constraints on the class probabilities and that it is the user's responsibility to ensure
          ``target`` contains valid probability distributions (see below examples section for more details).
        - Output: If reduction is 'none', shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.

        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}

    Examples:

        >>> # Example of target with class indices
        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()

    .. note::
        When ``target`` contains class probabilities, it should consist of soft labels—that is,
        each ``target`` entry should represent a probability distribution over the possible classes for a given data sample,
        with individual probabilities between ``[0,1]`` and the total distribution summing to 1.
        This is why the :func:`softmax()` function is applied to the ``target`` in the class probabilities example above.

        PyTorch does not validate whether the values provided in ``target`` lie in the range ``[0,1]``
        or whether the distribution of each data sample sums to ``1``.
        No warning will be raised and it is the user's responsibility
        to ensure that ``target`` contains valid probability distributions.
        Providing arbitrary values may yield misleading loss values and unstable gradients during training.

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example of target with incorrectly specified class probabilities
        >>> loss = nn.CrossEntropyLoss()
        >>> torch.manual_seed(283)
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> # Provided target class probabilities are not in range [0,1]
        >>> target
        tensor([[ 0.7105,  0.4446,  2.0297,  0.2671, -0.6075],
                [-1.0496, -0.2753, -0.3586,  0.9270,  1.0027],
                [ 0.7551,  0.1003,  1.3468, -0.3581, -0.9569]])
        >>> # Provided target class probabilities do not sum to 1
        >>> target.sum(axis=1)
        tensor([2.8444, 0.2462, 0.8873])
        >>> # No error message and possible misleading loss value
        >>> loss(input, target).item()
        4.6379876136779785
        >>>
        >>> # Example of target with correctly specified class probabilities
        >>> # Use .softmax() to ensure true probability distribution
        >>> target_new = target.softmax(dim=1)
        >>> # New target class probabilities all in range [0,1]
        >>> target_new
        tensor([[0.1559, 0.1195, 0.5830, 0.1000, 0.0417],
                [0.0496, 0.1075, 0.0990, 0.3579, 0.3860],
                [0.2607, 0.1355, 0.4711, 0.0856, 0.0471]])
        >>> # New target class probabilities sum to 1
        >>> target_new.sum(axis=1)
        tensor([1.0000, 1.0000, 1.0000])
        >>> loss(input, target_new).item()
        2.55349063873291
    """

    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    ignore_index: int
    label_smoothing: float

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

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

        self.features = self.build_features()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.avgpool = F.adaptive_avg_pool2d
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
        x = self.avgpool(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = self.cross_entropy_loss.to(self.device)
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
