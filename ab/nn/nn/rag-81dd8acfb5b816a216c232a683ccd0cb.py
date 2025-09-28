# Auto-generated single-file for GroupAll
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional

# ---- original imports from contributing modules ----
from torch import nn as nn

# ---- GroupAll (target) ----
class GroupAll(nn.Module):
    """Group xyz with feature.

    Args:
        use_xyz (bool): Whether to use xyz.
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self,
                xyz: torch.Tensor,
                new_xyz: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            new_xyz (Tensor): new xyz coordinates of the features.
            features (Tensor): (B, C, N) features to group.

        Returns:
            Tensor: (B, C + 3, 1, N) Grouped feature.
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                # (B, 3 + C, 1, N)
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features

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
        self.group_all = GroupAll(use_xyz=True)
        self.classifier = nn.Linear(35, self.num_classes)

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
        B, C, H, W = x.shape
        xyz = torch.randn(B, H * W, 3, device=x.device)
        new_xyz = torch.randn(B, H * W, 3, device=x.device)
        features = x.view(B, C, H * W)
        grouped = self.group_all(xyz, new_xyz, features)
        x = grouped.view(B, -1, H * W).mean(dim=2)
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
