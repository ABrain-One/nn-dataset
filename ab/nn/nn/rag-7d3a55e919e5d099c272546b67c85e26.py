# Auto-generated single-file for AdaptiveWingLoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- original imports from contributing modules ----

# ---- AdaptiveWingLoss (target) ----
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape +
                                                 (1, ) * ndim_pad)
            loss = self.criterion(output * target_weights,
                                  target * target_weights)
        else:
            loss = self.criterion(output, target)

        return loss


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
