import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            # Trivial Swish -> pointwise
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            # Inverted Residual
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            # Trivial Swish -> pointwise
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            InvertedResidual(32, 32, 1),
            nn.MaxPool2d(2, 2),
            InvertedResidual(32, 64, 2),
            nn.MaxPool2d(2, 2),
            InvertedResidual(64, 128, 2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self._set_hyper(prm)

    def _set_hyper(self, prm):
        self.lr = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()