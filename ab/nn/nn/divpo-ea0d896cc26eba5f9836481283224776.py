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
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = self._make_block(32, 32, 1)
        self.block2 = self._make_block(32, 64, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def _make_block(self, in_channels, out_channels, blocks):
        layers = [nn.BatchNorm2d(in_channels)]
        layers.append(self._make_layer(in_channels, out_channels, blocks))
        return nn.Sequential(*layers)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        stride = 1
        layers.append(self._make_basic_block(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layers.append(self._make_basic_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _make_basic_block(self, in_channels, out_channels, stride):
        expansion = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=3, stride=1, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels * expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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