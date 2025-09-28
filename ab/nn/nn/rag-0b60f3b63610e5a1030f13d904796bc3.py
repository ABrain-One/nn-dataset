# Auto-generated single-file for HuasdorffDisstanceLoss
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import *
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import pickle as load
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmseg.models.losses.utils.get_class_weight ----
def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = load(class_weight)

    return class_weight

# ---- mmseg.models.losses.huasdorff_distance_loss.compute_dtm ----
def compute_dtm(img_gt: Tensor, pred: Tensor) -> Tensor:
    """
    compute the distance transform map of foreground in mask
    Args:
        img_gt: Ground truth of the image, (b, h, w)
        pred: Predictions of the segmentation head after softmax, (b, c, h, w)

    Returns:
        output: the foreground Distance Map (SDM)
        dtm(x) = 0; x in segmentation boundary
                inf|x-y|; x in segmentation
    """

    fg_dtm = torch.zeros_like(pred)
    out_shape = pred.shape
    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):  # default 0 channel is background
            posmask = img_gt[b].byte()
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = torch.from_numpy(posdis)

    return fg_dtm

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

# ---- mmseg.models.losses.utils.weighted_loss ----
def weighted_loss(loss_func):
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

    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

# ---- mmseg.models.losses.huasdorff_distance_loss.hd_loss ----
def hd_loss(seg_soft: Tensor,
            gt: Tensor,
            seg_dtm: Tensor,
            gt_dtm: Tensor,
            class_weight=None,
            ignore_index=255) -> Tensor:
    """
    compute huasdorff distance loss for segmentation
    Args:
        seg_soft: softmax results, shape=(b,c,x,y)
        gt: ground truth, shape=(b,x,y)
        seg_dtm: segmentation distance transform map, shape=(b,c,x,y)
        gt_dtm: ground truth distance transform map, shape=(b,c,x,y)

    Returns:
        output: hd_loss
    """
    assert seg_soft.shape[0] == gt.shape[0]
    total_loss = 0
    num_class = seg_soft.shape[1]
    if class_weight is not None:
        assert class_weight.ndim == num_class
    for i in range(1, num_class):
        if i != ignore_index:
            delta_s = (seg_soft[:, i, ...] - gt.float())**2
            s_dtm = seg_dtm[:, i, ...]**2
            g_dtm = gt_dtm[:, i, ...]**2
            dtm = s_dtm + g_dtm
            multiplied = torch.einsum('bxy, bxy->bxy', delta_s, dtm)
            hd_loss = multiplied.mean()
        if class_weight is not None:
            hd_loss *= class_weight[i]
        total_loss += hd_loss

    return total_loss / num_class

# ---- HuasdorffDisstanceLoss (target) ----
class HuasdorffDisstanceLoss(nn.Module):
    """HuasdorffDisstanceLoss. This loss is proposed in `How Distance Transform
    Maps Boost Segmentation CNNs: An Empirical Study.

    <http://proceedings.mlr.press/v121/ma20b.html>`_.
    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_huasdorff_disstance',
                 **kwargs):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(self,
                pred: Tensor,
                target: Tensor,
                avg_factor=None,
                reduction_override=None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predictions of the segmentation head. (B, C, H, W)
            target (Tensor): Ground truth of the image. (B, H, W)
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred_soft = F.softmax(pred, dim=1)
        valid_mask = (target != self.ignore_index).long()
        target = target * valid_mask

        with torch.no_grad():
            gt_dtm = compute_dtm(target.cpu(), pred_soft)
            gt_dtm = gt_dtm.float()
            seg_dtm2 = compute_dtm(
                pred_soft.argmax(dim=1, keepdim=False).cpu(), pred_soft)
            seg_dtm2 = seg_dtm2.float()

        loss_hd = self.loss_weight * hd_loss(
            pred_soft,
            target,
            seg_dtm=seg_dtm2,
            gt_dtm=gt_dtm,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss_hd

    def loss_name(self):
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
        self.huasdorff_loss = HuasdorffDisstanceLoss(reduction='mean', loss_weight=1.0, ignore_index=255)
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
