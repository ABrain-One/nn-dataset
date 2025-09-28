# Auto-generated single-file for GaussianFocalLoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Callable
from typing import Union
from typing import Optional
from typing import Callable
from functools import *
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmdet.models.losses.utils.reduce_loss ----
def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

# ---- mmdet.models.losses.utils.weight_reduce_loss ----
def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

# ---- mmdet.models.losses.gaussian_focal_loss.gaussian_focal_loss_with_pos_inds ----
def gaussian_focal_loss_with_pos_inds(
        pred: Tensor,
        gaussian_target: Tensor,
        pos_inds: Tensor,
        pos_labels: Tensor,
        alpha: float = 2.0,
        gamma: float = 4.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = 'mean',
        avg_factor: Optional[Union[int, float]] = None) -> Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Note: The index with a value of 1 in ``gaussian_target`` in the
    ``gaussian_focal_loss`` function is a positive sample, but in
    ``gaussian_focal_loss_with_pos_inds`` the positive sample is passed
    in through the ``pos_inds`` parameter.

    Args:
        pred (torch.Tensor): The prediction. The shape is (N, num_classes).
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution. The shape is (N, num_classes).
        pos_inds (torch.Tensor): The positive sample index.
            The shape is (M, ).
        pos_labels (torch.Tensor): The label corresponding to the positive
            sample index. The shape is (M, ).
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to 'mean`.
        avg_factor (int, float, optional): Average factor that is used to
            average the loss. Defaults to None.
    """
    eps = 1e-12
    neg_weights = (1 - gaussian_target).pow(gamma)

    pos_pred_pix = pred[pos_inds]
    pos_pred = pos_pred_pix.gather(1, pos_labels.unsqueeze(1))
    pos_loss = -(pos_pred + eps).log() * (1 - pos_pred).pow(alpha)
    pos_loss = weight_reduce_loss(pos_loss, None, reduction, avg_factor)

    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    neg_loss = weight_reduce_loss(neg_loss, None, reduction, avg_factor)

    return pos_weight * pos_loss + neg_weight * neg_loss

# ---- mmdet.models.losses.utils.weighted_loss ----
def weighted_loss(loss_func: Callable) -> Callable:
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    def wrapper(pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                reduction: str = 'mean',
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): Options are "none", "mean" and "sum".
                Defaults to 'mean'.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

# ---- mmdet.models.losses.gaussian_focal_loss.gaussian_focal_loss ----
def gaussian_focal_loss(pred: Tensor,
                        gaussian_target: Tensor,
                        alpha: float = 2.0,
                        gamma: float = 4.0,
                        pos_weight: float = 1.0,
                        neg_weight: float = 1.0) -> Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_weight * pos_loss + neg_weight * neg_loss

# ---- GaussianFocalLoss (target) ----
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(self,
                 alpha: float = 2.0,
                 gamma: float = 4.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                pos_inds: Optional[Tensor] = None,
                pos_labels: Optional[Tensor] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[int, float]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if pos_inds is not None:
            assert pos_labels is not None
            # Only used by centernet update version
            loss_reg = self.loss_weight * gaussian_focal_loss_with_pos_inds(
                pred,
                target,
                pos_inds,
                pos_labels,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_reg = self.loss_weight * gaussian_focal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor)
        return loss_reg

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
        self.gaussian_focal_loss = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction='mean', loss_weight=1.0, pos_weight=1.0, neg_weight=1.0)
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
