# Auto-generated single-file for ConvBN
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- ConvBN (target) ----
class ConvBN(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            with_bn: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_channels))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

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

    def build_features(self):
        layers = []
        layers += [
            ConvBN(self.in_channels, 32, kernel_size=3, stride=1, padding=1, with_bn=True),
            nn.ReLU(inplace=True),
            ConvBN(32, 32, kernel_size=3, stride=1, padding=1, with_bn=True),
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

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
