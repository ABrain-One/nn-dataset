import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expand_ratio, momentum=0.99))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=in_channels * expand_ratio,
                out_channels=in_channels * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expand_ratio,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expand_ratio, momentum=0.99))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=in_channels * expand_ratio,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.99))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3_Small(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNetV3_Small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            InvertedResidual(16, 16, 1, 3),
            InvertedResidual(16, 24, 2, 3),
            InvertedResidual(24, 24, 1, 3),
            InvertedResidual(24, 32, 2, 3),
            InvertedResidual(32, 32, 1, 3),
            InvertedResidual(32, 64, 2, 3),
            InvertedResidual(64, 64, 1, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.model = MobileNetV3_Small(in_channels, num_classes)
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=0.0,
            nesterov=True
        )

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criteria[0](output, y)
            loss.backward()
            self.optimizer.step()