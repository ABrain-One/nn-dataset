# Auto-generated single-file for MaskedDiceLoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- original imports from contributing modules ----

# ---- MaskedDiceLoss (target) ----
class MaskedDiceLoss(nn.Module):
    """Masked dice loss.

    Args:
        eps (float, optional): Eps to avoid zero-divison error.  Defaults to
            1e-6.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction in any shape.
            gt (torch.Tensor): The learning target of the prediction in the
                same shape as pred.
            mask (torch.Tensor, optional): Binary mask in the same shape of
                pred, indicating positive regions to calculate the loss. Whole
                region will be taken into account if not provided. Defaults to
                None.

        Returns:
            torch.Tensor: The loss value.
        """

        assert pred.size() == gt.size() and gt.numel() > 0
        if mask is None:
            mask = torch.ones_like(gt)
        assert mask.size() == gt.size()

        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        mask = mask.contiguous().view(mask.size(0), -1)
        pred = pred * mask
        gt = gt * mask

        dice_coeff = (2 * (pred * gt).sum()) / (
            pred.sum() + gt.sum() + self.eps)

        return 1 - dice_coeff

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
        self.masked_dice_loss = MaskedDiceLoss(eps=1e-6)
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
        
        pred = torch.randn_like(x)
        gt = torch.randint(0, 2, (x.size(0), x.size(1)), dtype=torch.float, device=x.device)
        mask = torch.ones_like(gt)
        
        loss_value = self.masked_dice_loss(pred, gt, mask)
        
        loss_feature = loss_value.unsqueeze(-1).expand(x.size(0), x.size(1))
        x_combined = x + loss_feature * 0.1
        
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
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
