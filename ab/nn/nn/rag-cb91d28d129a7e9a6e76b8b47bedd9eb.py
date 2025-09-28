# Auto-generated single-file for WindowDepartition
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor

# ---- original imports from contributing modules ----
from torch import nn, Tensor

# ---- WindowDepartition (target) ----
class WindowDepartition(nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, h_partitions: int, w_partitions: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)
        return x

def supported_hyperparameters():
    return ['lr', 'momentum']

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
        
        self.window_departition = WindowDepartition()
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
        p = 2  # partition size
        h_partitions = H // p
        w_partitions = W // p
        
        x = x.view(B, C, h_partitions, p, w_partitions, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, h_partitions * w_partitions, p * p, C)
        
        x = self.window_departition(x, p, h_partitions, w_partitions)
        x = x.mean(dim=[2, 3])
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
