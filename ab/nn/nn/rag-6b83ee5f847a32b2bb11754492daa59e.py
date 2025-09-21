# Auto-generated single-file for DownsampleNormFirst
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
from torch.nn import LayerNorm
import torch

# ---- original imports from contributing modules ----
from torch import nn

# ---- DownsampleNormFirst (target) ----
class DownsampleNormFirst(nn.Module):

    def __init__(
            self,
            in_chs=96,
            out_chs=198,
            norm_layer=LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(in_chs)
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x

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
        
        self.downsample_norm_first = DownsampleNormFirst(
            in_chs=32,
            out_chs=32,
            norm_layer=LayerNorm
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
