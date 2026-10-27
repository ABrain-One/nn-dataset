import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Derive from in_shape for edge-friendly channels-first (C, H, W)
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Use in_channels from in_shape to ensure compatibility with input shape
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.to(device)
        self._make_criterion(prm)
        self._make_optimizer(prm)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.criteria = self.criterion
        self.optimizer = self._get_optimizer(prm)
        self.to(self.device)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def _make_criterion(self, prm):
        if prm is None:
            prm = {}
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def _make_optimizer(self, prm):
        if prm is None:
            prm = {}
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def _get_optimizer(self, prm):
        if prm is None:
            prm = {}
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, num_classes={self.num_classes}, dropout={self.dropout})"