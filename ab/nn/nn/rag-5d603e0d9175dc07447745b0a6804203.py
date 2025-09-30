# Auto-generated single-file for KeypointOHKMMSELoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- original imports from contributing modules ----

# ---- KeypointOHKMMSELoss (target) ----
class KeypointOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        topk (int): Only top k joint losses are kept. Defaults to 8
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 topk: int = 8,
                 loss_weight: float = 1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: Tensor) -> Tensor:
        """Online hard keypoint mining.

        Note:
            - batch_size: B
            - num_keypoints: K

        Args:
            loss (Tensor): The losses with shape [B, K]

        Returns:
            Tensor: The calculated loss.
        """
        ohkm_loss = 0.
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not be '
                             f'larger than num_keypoints ({num_keypoints}).')

        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(
                    self.criterion(output[:, idx] * target_weight,
                                   target[:, idx] * target_weight))
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))

        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight

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
        self.keypoint_ohkm_mse_loss = KeypointOHKMMSELoss(use_target_weight=True, topk=8, loss_weight=1.0)
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
        
        output_heatmaps = torch.randn(x.size(0), 17, 8, 8, device=x.device)
        target_heatmaps = torch.randn(x.size(0), 17, 8, 8, device=x.device)
        target_weights = torch.ones(x.size(0), 17, device=x.device)
        
        ohkm_mse_loss = self.keypoint_ohkm_mse_loss(output_heatmaps, target_heatmaps, target_weights)
        x_combined = x + ohkm_mse_loss.unsqueeze(-1)
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
