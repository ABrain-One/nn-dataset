import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == out_channels

        layers = []
        layers += [
            conv_1x1_bn(input_channels, out_channels, stride=1, padding=0, groups=1),
            InvertedResidualBlock(out_channels, out_channels, expand_ratio=expand_ratio)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio):
        super(InvertedResidualBlock, self).__init__()

        self.use_res = expand_ratio == 1 and in_channels == out_channels

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=1, padding=1, groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return self.conv(x)
        else:
            return x + self.conv(x)

def conv_1x1_bn(in_channels, out_channels, stride, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        self.inverted_residual_1 = InvertedResidual(32, 16, 1, 1)
        self.inverted_residual_2 = InvertedResidual(16, 24, 1, 1)
        self.inverted_residual_3 = InvertedResidual(24, 32, 1, 1)
        self.inverted_residual_4 = InvertedResidual(32, 64, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.inverted_residual_1(x)
        x = self.inverted_residual_2(x)
        x = self.inverted_residual_3(x)
        x = self.inverted_residual_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = self.build_features(in_shape, 10)
        self.inverted_residual = InvertedResidualBlock(32, 32, 1)
        self.classifier = nn.Linear(32, out_shape[0])
        self.hyperparameters = prm

    def build_features(self, in_shape, out_dim):
        layers = [
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.inverted_residual(x)
        return self.classifier(x.mean(dim=(2, 3)))

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()