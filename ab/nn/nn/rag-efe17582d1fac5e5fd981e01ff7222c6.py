# Auto-generated single-file for SiLogLoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union
from typing import Union
from typing import Optional
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmseg.models.losses.utils.reduce_loss ----
def reduce_loss(loss, reduction) -> torch.Tensor:
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

# ---- mmseg.models.losses.utils.weight_reduce_loss ----
def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None) -> torch.Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
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

# ---- mmseg.models.losses.silog_loss.silog_loss ----
def silog_loss(pred: Tensor,
               target: Tensor,
               weight: Optional[Tensor] = None,
               eps: float = 1e-4,
               reduction: Union[str, None] = 'mean',
               avg_factor: Optional[int] = None) -> Tensor:
    """Computes the Scale-Invariant Logarithmic (SI-Log) loss between
    prediction and target.

    Args:
        pred (Tensor): Predicted output.
        target (Tensor): Ground truth.
        weight (Optional[Tensor]): Optional weight to apply on the loss.
        eps (float): Epsilon value to avoid division and log(0).
        reduction (Union[str, None]): Specifies the reduction to apply to the
            output: 'mean', 'sum' or None.
        avg_factor (Optional[int]): Optional average factor for the loss.

    Returns:
        Tensor: The calculated SI-Log loss.
    """
    pred, target = pred.flatten(1), target.flatten(1)
    valid_mask = (target > eps).detach().float()

    diff_log = torch.log(target.clamp(min=eps)) - torch.log(
        pred.clamp(min=eps))

    valid_mask = (target > eps).detach() & (~torch.isnan(diff_log))
    diff_log[~valid_mask] = 0.0
    valid_mask = valid_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(
        dim=1) / valid_mask.sum(dim=1).clamp(min=eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1).clamp(min=eps)

    loss = torch.sqrt(diff_log_sq_mean - 0.5 * diff_log_mean.pow(2))

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

# ---- SiLogLoss (target) ----
class SiLogLoss(nn.Module):
    """Compute SiLog loss.

    Args:
        reduction (str, optional): The method used
            to reduce the loss. Options are "none",
            "mean" and "sum". Defaults to 'mean'.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        eps (float): Avoid dividing by zero. Defaults to 1e-3.
        loss_name (str, optional): Name of the loss item. If you want this
            loss item to be included into the backward graph, `loss_` must
            be the prefix of the name. Defaults to 'loss_silog'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-6,
                 loss_name='loss_silog'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):

        assert pred.shape == target.shape, 'the shapes of pred ' \
            f'({pred.shape}) and target ({target.shape}) are mismatch'

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * silog_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
        )

        return loss

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

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
        self.silog_loss = SiLogLoss(reduction='mean', loss_weight=1.0, eps=1e-6, loss_name='loss_silog')
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            # Use both CrossEntropyLoss and SiLogLoss for demonstration
            ce_loss = self.criterion(output, target)
            # Create dummy depth prediction for SiLogLoss demonstration
            depth_pred = torch.randn_like(output)
            depth_target = torch.randn_like(output)
            silog_loss = self.silog_loss(depth_pred, depth_target)
            total_loss = ce_loss + silog_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
