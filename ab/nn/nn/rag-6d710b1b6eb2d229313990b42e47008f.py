# Auto-generated single-file for DyReLU
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor

# ---- original imports from contributing modules ----

# ---- DyReLU (target) ----
class DyReLU(nn.Module):
    """Dynamic ReLU."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expand_ratio = expand_ratio
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // expand_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // expand_ratio,
                      out_channels * self.expand_ratio),
            nn.Hardsigmoid(inplace=True))

    def forward(self, x) -> Tensor:
        x_out = x
        b, c, h, w = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x).view(b, -1, 1, 1)

        a1, b1, a2, b2 = torch.split(x, self.out_channels, dim=1)
        a1 = (a1 - 0.5) * 2 + 1.0
        a2 = (a2 - 0.5) * 2
        b1 = b1 - 0.5
        b2 = b2 - 0.5
        out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        return out

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        self.dyrelu = DyReLU(
            in_channels=32,
            out_channels=32,
            expand_ratio=4
        )

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.dyrelu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
