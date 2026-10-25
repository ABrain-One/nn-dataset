import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # Pointwise convolution
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(32, 32, 3, 1)
        self.layer2 = self._make_layer(32, 32, 3, 2)
        self.layer3 = self._make_layer(32, 64, 3, 2, expand_ratio=6)
        self.layer4 = self._make_layer(64, 64, 3, 1, expand_ratio=6)
        self.layer5 = self._make_layer(64, 96, 5, 2, expand_ratio=6)
        self.layer6 = self._make_layer(96, 96, 3, 1, expand_ratio=6)
        self.layer7 = self._make_layer(96, 192, 3, 1, expand_ratio=6)

        self.conv = nn.Sequential(
            nn.Conv2d(192, 1280, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride, expand_ratio=1):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio))
        for _ in range(1, blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
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