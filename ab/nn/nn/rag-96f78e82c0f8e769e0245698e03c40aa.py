# Auto-generated single-file for GHMC
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmdet.models.losses.ghm_loss._expand_onehot_labels ----
def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

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

# ---- GHMC (target) ----
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to "mean"
    """

    def __init__(self,
                 bins=10,
                 momentum=0,
                 use_sigmoid=True,
                 loss_weight=1.0,
                 reduction='mean'):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                pred,
                target,
                label_weight,
                reduction_override=None,
                **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(
                target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        loss = weight_reduce_loss(
            loss, weights, reduction=reduction, avg_factor=tot)
        return loss * self.loss_weight

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
        self.classifier = nn.Linear(32, self.num_classes)
        self.ghmc_loss = GHMC(bins=10, momentum=0.75, use_sigmoid=True, loss_weight=1.0)

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
