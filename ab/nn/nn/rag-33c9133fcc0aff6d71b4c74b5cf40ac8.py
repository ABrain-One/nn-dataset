# Auto-generated single-file for OhemCrossEntropy
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Union
from typing import List
import torch
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass
from typing import Union
from typing import Optional

# ---- original imports from contributing modules ----

# ---- OhemCrossEntropy (target) ----
class OhemCrossEntropy(nn.Module):
    """OhemCrossEntropy loss.

    This func is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L43>`_.  # noqa

    Licensed under the MIT License.

    Args:
        ignore_label (int): Labels to ignore when computing the loss.
            Default: 255
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: 0.7.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 ignore_label: int = 255,
                 thres: float = 0.7,
                 min_kept: int = 100000,
                 loss_weight: float = 1.0,
                 class_weight: Optional[Union[List[float], str]] = None,
                 loss_name: str = 'loss_ohem'):
        super().__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.class_weight = class_weight

    def forward(self, score: Tensor, target: Tensor) -> Tensor:
        """Forward function.
        Args:
            score (Tensor): Predictions of the segmentation head.
            target (Tensor): Ground truth of the image.

        Returns:
            Tensor: Loss tensor.
        """
        # score: (N, C, H, W)
        pred = F.softmax(score, dim=1)
        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
        else:
            class_weight = None

        pixel_losses = F.cross_entropy(
            score,
            target,
            weight=class_weight,
            ignore_index=self.ignore_label,
            reduction='none').contiguous().view(-1)  # (N*H*W)
        mask = target.contiguous().view(-1) != self.ignore_label  # (N*H*W)

        tmp_target = target.clone()  # (N, H, W)
        tmp_target[tmp_target == self.ignore_label] = 0
        # pred: (N, C, H, W) -> (N*H*W, C)
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        # pred: (N*H*W, C) -> (N*H*W), ind: (N*H*W)
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        if pred.numel() > 0:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        else:
            return score.new_tensor(0.0)
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return self.loss_weight * pixel_losses.mean()

    def loss_name(self):
        return self.loss_name_

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
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = self.features(x)
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
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
