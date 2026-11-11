import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1]) if len(in_shape) == 4 else int(in_shape[0])
        num_classes = int(out_shape[0])

        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32, 2, 2)
        self.classifier = nn.Linear(32, num_classes)

        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layers(self, out_channels, blocks, stride):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2, stride=stride))
        layers.append(self._make_layer(out_channels, blocks))
        return nn.Sequential(*layers)

    def _make_layer(self, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(self.BasicBlock(out_channels))
        return nn.Sequential(*layers)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, out_channels):
            super(Net.BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = nn.functional.relu(out, inplace=True)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity
            out = nn.functional.relu(out, inplace=True)

            return out

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.layers(x)
        x = nn.functional.avg_pool2d(x, x.size()[3])
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