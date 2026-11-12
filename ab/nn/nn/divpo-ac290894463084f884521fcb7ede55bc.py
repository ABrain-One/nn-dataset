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
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            self._make_layer(32, 32, 1),
            self._make_layer(32, 64, 2),
            self._make_layer(64, 64, 1)
        )
        self.classifier = nn.Linear(64, self.num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layer(self, in_planes, planes, num_blocks):
        layers = []
        stride = 2 if num_blocks > 0 else 1
        layers.append(self._make_pad(stride))
        layers.append(nn.ReLU(inplace=True))
        layers.append(self._make_conv(in_planes, planes, stride))
        while num_blocks > 0:
            layers.append(self._make_conv(planes, planes, 1))
            num_blocks -= 1
        return nn.Sequential(*layers)

    def _make_pad(self, stride):
        if stride == 2:
            return nn.AvgPool2d(2)
        return nn.Identity()

    def _make_conv(self, in_planes, planes, stride):
        return nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        return self.classifier(self.stem(x).mean(2).mean(2))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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