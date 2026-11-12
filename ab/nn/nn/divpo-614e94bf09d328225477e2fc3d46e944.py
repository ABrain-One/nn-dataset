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
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.conv2 = self._make_layer(32, 32, 1)
        self.conv3 = self._make_layer(32, 64, 2)
        self.conv4 = self._make_layer(64, 128, 2)
        self.conv5 = self._make_layer(128, 256, 2)
        self.ap = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(256 * 6 * 6, 10)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for block in range(blocks):
            first = (in_channels != out_channels) and block == 0
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
              padding=1, bias=False if first else True, dilation=1,
              groups=1))
            if first:
                layers.extend([nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True)])
                in_channels = out_channels
            elif block == 0:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
              padding=1, bias=False, groups=1, dilation=1,
              stride=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ap(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()