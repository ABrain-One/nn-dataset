import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1]) if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layers = self._make_layers(32, 2, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def _make_layers(self, in_channels, blocks, out_channels):
        layers = []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            layers += [nn.BatchNorm2d(out_channels)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layers(x)
        x = self.avgpool(x)
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