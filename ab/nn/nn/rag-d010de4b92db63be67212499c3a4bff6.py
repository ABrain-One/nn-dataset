# Auto-generated single-file for SpatialGatingUnit
# Dependencies are emitted in topological order (utilities first).
from collections.abc import Sequence
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- original imports from contributing modules ----

# ---- SpatialGatingUnit (target) ----
class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit.

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim: int, seq_len: int, norm_layer: type = nn.LayerNorm) -> None:
        """Initialize Spatial Gating Unit.

        Args:
            dim: Dimension of input features.
            seq_len: Sequence length.
            norm_layer: Normalization layer.
        """
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self) -> None:
        """Initialize weights for projection gate."""
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial gating."""
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)

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
        self.spatial_gating = SpatialGatingUnit(dim=128, seq_len=64, norm_layer=nn.LayerNorm)
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
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_expanded = torch.cat([x_flat, x_flat], dim=-1)  
        x_gated = self.spatial_gating(x_expanded)  
        x = x_gated[:, :, :C].transpose(1, 2).view(B, C, H, W)  
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
