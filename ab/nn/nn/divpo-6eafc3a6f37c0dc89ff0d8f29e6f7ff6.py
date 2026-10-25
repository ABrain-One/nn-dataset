import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, 1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels * expansion_factor,
                in_channels * expansion_factor,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion_factor,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(in_channels * expansion_factor, out_channels, 1, 1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

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
        layers = []
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # EfficientNet-inspired inverted residual blocks
        inverted_residual_block = InvertedResidual(32, 32, 1, 3)
        layers += [inverted_residual_block for _ in range(3)]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, 1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels * expansion_factor,
                in_channels * expansion_factor,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion_factor,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(in_channels * expansion_factor, out_channels, 1, 1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

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
        layers = []
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # EfficientNet-inspired inverted residual blocks
        inverted_residual_block = InvertedResidual(32, 32, 1, 3)
        layers += [inverted_residual_block for _ in range(3)]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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