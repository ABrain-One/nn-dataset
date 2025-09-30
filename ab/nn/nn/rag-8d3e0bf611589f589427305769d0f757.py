# Auto-generated single-file for BoundaryLoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- original imports from contributing modules ----

# ---- BoundaryLoss (target) ----
class BoundaryLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.

    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, bd_pre: Tensor, bd_gt: Tensor) -> Tensor:
        """Forward function.
        Args:
            bd_pre (Tensor): Predictions of the boundary head.
            bd_gt (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = bd_gt.view(1, -1).float()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, reduction='mean')

        return self.loss_weight * loss

    def loss_name(self):
        return self.loss_name_


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self.boundary_head = torch.nn.Conv2d(32, 1, kernel_size=1)

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        boundary_pred = self.boundary_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.boundary_loss = BoundaryLoss(loss_weight=1.0, loss_name='loss_boundary').to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            
            # Create dummy boundary ground truth for demonstration
            boundary_gt = torch.zeros(inputs.size(0), 1, inputs.size(2)//4, inputs.size(3)//4, device=self.device)
            
            # Get boundary prediction from the boundary head
            x = self.features(inputs)
            boundary_pred = self.boundary_head(x)
            
            # Resize boundary prediction to match ground truth
            boundary_pred = torch.nn.functional.interpolate(boundary_pred, size=boundary_gt.shape[2:], mode='bilinear', align_corners=False)
            
            # Calculate losses
            classification_loss = self.criteria(outputs, labels)
            boundary_loss = self.boundary_loss(boundary_pred, boundary_gt)
            total_loss = classification_loss + boundary_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
