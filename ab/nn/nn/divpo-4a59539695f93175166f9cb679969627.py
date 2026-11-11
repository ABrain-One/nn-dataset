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
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._create_layer(32, 32, 1)
        self.layer2 = self._create_layer(32, 64, 1)
        self.layer3 = self._create_layer(64, 128, 2)
        self.layer4 = self._create_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(256 * 7 * 7, self.num_classes)

    def _create_layer(self, in_c, out_c, blocks):
        layer = []
        layer += [nn.Conv2d(in_c, out_c, kernel_size=3,
                            padding=1, bias=False)]
        layer += [nn.BatchNorm2d(out_c)]
        layer += [nn.ReLU(inplace=True)]
        for i in range(1, blocks):
            layer += [nn.Conv2d(out_c, out_c, kernel_size=3,
                                padding=1, bias=False)]
            layer += [nn.BatchNorm2d(out_c)]
            layer += [nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()