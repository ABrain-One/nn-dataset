# Auto-generated single-file for PatchDropout
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional, Tuple
from typing import Tuple
from typing import Optional

# ---- timm.layers.patch_dropout.patch_dropout_forward ----
def patch_dropout_forward(
        x: torch.Tensor,
        prob: float,
        num_prefix_tokens: int,
        ordered: bool,
        training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Common forward logic for patch dropout.

    Args:
        x: Input tensor of shape (B, L, D)
        prob: Dropout probability
        num_prefix_tokens: Number of prefix tokens to preserve
        ordered: Whether to maintain patch order
        training: Whether in training mode

    Returns:
        Tuple of (output tensor, keep_indices or None)
    """
    if not training or prob == 0.:
        return x, None

    if num_prefix_tokens:
        prefix_tokens, x = x[:, :num_prefix_tokens], x[:, num_prefix_tokens:]
    else:
        prefix_tokens = None

    B = x.shape[0]
    L = x.shape[1]
    num_keep = max(1, int(L * (1. - prob)))
    keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]

    if ordered:
        # NOTE does not need to maintain patch order in typical transformer use,
        # but possibly useful for debug / visualization
        keep_indices = keep_indices.sort(dim=-1)[0]

    x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

    if prefix_tokens is not None:
        x = torch.cat((prefix_tokens, x), dim=1)

    return x, keep_indices

# ---- PatchDropout (target) ----
class PatchDropout(nn.Module):
    """
    Patch Dropout without returning indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = patch_dropout_forward(
            x,
            self.prob,
            self.num_prefix_tokens,
            self.ordered,
            self.training
        )
        return output

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.patch_dropout = PatchDropout(prob=0.3, num_prefix_tokens=0, ordered=False)
        
        self.classifier = nn.Linear(32, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        
        x = x.view(B, C, H*W).transpose(1, 2)
        x = self.patch_dropout(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
    
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
