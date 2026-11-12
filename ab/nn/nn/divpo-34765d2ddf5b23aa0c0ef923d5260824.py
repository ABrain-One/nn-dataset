import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.features)
        self.conv2 = self._make_layer(32, 32, 1)
        self.conv3 = self._make_layer(32, 64, 2)
        self.conv4 = self._make_layer(64, 128, 2)
        self.conv5 = self._make_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(256 * 7 * 7, self.num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layer = []
        for i in range(blocks):
            stride = 1
            if i == 0:
                stride = 2
            layer += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            in_channels = out_channels
        return nn.Sequential(*layer)

    def features_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.features_forward(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                          weight_decay=1e-4, nesterov=True)

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