# Auto-generated single-file for _ReplicationPadNd
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from collections.abc import Sequence
import torch.nn as nn
import torch

# ---- original imports from contributing modules ----

# ---- _ReplicationPadNd (target) ----
class _ReplicationPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "replicate")

    def extra_repr(self) -> str:
        return f"{self.padding}"

def supported_hyperparameters():
    return ['lr', 'momentum']

class Net(Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        
        self.replication_pad = _ReplicationPadNd()
        self.replication_pad.padding = [1, 1, 1, 1]
        self.classifier = nn.Linear(64 * 6 * 6, self.num_classes)

    def build_features(self):
        import torch.nn as nn
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
        
        x = F.adaptive_avg_pool2d(x, (4, 4))
        B, C, H, W = x.shape
        
        x = self.replication_pad(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        import torch
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = F.cross_entropy

    def learn(self, data_roll):
        import torch
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
