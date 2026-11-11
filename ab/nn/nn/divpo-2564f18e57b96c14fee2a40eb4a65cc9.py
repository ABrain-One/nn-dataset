import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.block1 = self._make_block(32, 32, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = self._make_block(32, 64, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(64 * 8 * 8, self.num_classes)

    def _make_block(self, in_channels, out_channels, stride, groups):
        expansion = 4
        width = int(out_channels * expansion)
        layers = [nn.Conv2d(in_channels, width, kernel_size=1, bias=False)]
        layers += [
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width,
                width,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
        ]
        layers += [
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, kernel_size=1, bias=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = nn.AdaptiveAvgPool2d((8, 8))(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()