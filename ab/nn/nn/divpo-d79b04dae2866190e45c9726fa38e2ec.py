import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1] if len(in_shape) == 4 else in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.avgpool = nn.AvgPool2d(2)
        self.features = nn.Sequential(
            self._make_layer(8, 20, 1),
            self._make_layer(20, 32, 1)
        )
        self.classifier = nn.Linear(32, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layer(self, in_planes, planes, num_blocks):
        layers = []
        layers.append(self._make_basic_block(in_planes, planes))
        for i in range(1, num_blocks):
            layers.append(self._make_basic_block(planes, planes))
        return nn.Sequential(*layers)

    def _make_basic_block(self, in_planes, planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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