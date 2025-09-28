# Auto-generated single-file for Heatmap1DHead
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from collections.abc import Sequence

# ---- mmengine.model.weight_init.normal_init ----
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# ---- mmpose.models.heads.heatmap_heads.internet_head.make_linear_layers ----
def make_linear_layers(feat_dims, relu_final=False):
    """Make linear layers."""
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or \
                (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# ---- Heatmap1DHead (target) ----
class Heatmap1DHead(nn.Module):
    """Heatmap1DHead is a sub-module of Interhand3DHead, and outputs 1D
    heatmaps.

    Args:
        in_channels (int): Number of input channels. Defaults to 2048.
        heatmap_size (int): Heatmap size. Defaults to 64.
        hidden_dims (Sequence[int]): Number of feature dimension of FC layers.
            Defaults to ``(512, )``.
    """

    def __init__(self,
                 in_channels: int = 2048,
                 heatmap_size: int = 64,
                 hidden_dims: Sequence[int] = (512, )):

        super().__init__()

        self.in_channels = in_channels
        self.heatmap_size = heatmap_size

        feature_dims = [in_channels, *hidden_dims, heatmap_size]
        self.fc = make_linear_layers(feature_dims, relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(
            self.heatmap_size, dtype=heatmap1d.dtype,
            device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self.fc(feats)
        x = self.soft_argmax_1d(x).view(-1, 1)
        return x

    def init_weights(self):
        """Initialize model weights."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

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
        self.heatmap_1d_head = Heatmap1DHead(in_channels=32, heatmap_size=32, hidden_dims=(128,))
        self.additional_features = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(33, self.num_classes)

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
        x_heatmap = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x_heatmap = x_heatmap.flatten(1)
        heatmap_out = self.heatmap_1d_head(x_heatmap)
        
        x_additional = self.additional_features(x)
        x_additional = nn.functional.adaptive_avg_pool2d(x_additional, (1, 1))
        x_additional = x_additional.flatten(1)
        
        x_combined = torch.cat([heatmap_out, x_additional], dim=1)
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
            self.optimizer.step()
