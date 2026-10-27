import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 32, groups=32)

        self.syncbn1 = nn.SyncBatchNorm(32, eps=1e-4, momentum=0.1, affine=False)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32)
        )
        self.layer3 = Block(32, 64, kernel_size=3, padding=1, stride=2)

        self.syncbn2 = nn.SyncBatchNorm(64, eps=1e-4, momentum=0.1, affine=False)
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64)
        )
        self.layer4 = Block(64, 64)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.stem = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.syncbn1,
            self.dwconv1,
            self.layer3,
            self.syncbn2,
            self.dwconv2,
            self.layer4,
            nn.Dropout(p=0.05),
        )

        self.classifier = nn.Linear(64, num_classes)

        self.to(device)
        self_parems = list(self.stem.parameters()) + list(self.classifier.parameters())
        self._optimizer = torch.optim.SGD(
            self_parems,
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self._optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self._optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self._optimizer.step()