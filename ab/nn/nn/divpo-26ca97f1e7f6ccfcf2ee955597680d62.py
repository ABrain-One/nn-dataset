import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = make_divisible(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
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
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.in_resolution = in_shape[2]
        self.out_channels = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512]

        self.conv1 = ConvBNReLU(self.in_channels, self.out_channels[0], kernel_size=3, stride=1)
        self.inverted_residuals = nn.Sequential(
            InvertedResidual(self.out_channels[0], self.out_channels[1], stride=1, expand_ratio=3),
            InvertedResidual(self.out_channels[1], self.out_channels[2], stride=2, expand_ratio=3),
            InvertedResidual(self.out_channels[2], self.out_channels[3], stride=1, expand_ratio=3),
            InvertedResidual(self.out_channels[3], self.out_channels[4], stride=2, expand_ratio=3),
            InvertedResidual(self.out_channels[4], self.out_channels[5], stride=1, expand_ratio=3),
            InvertedResidual(self.out_channels[5], self.out_channels[6], stride=2, expand_ratio=3),
            InvertedResidual(self.out_channels[6], self.out_channels[7], stride=1, expand_ratio=3),
            InvertedResidual(self.out_channels[7], self.out_channels[8], stride=2, expand_ratio=3),
            InvertedResidual(self.out_channels[8], self.out_channels[9], stride=1, expand_ratio=3),
        )
        self.conv2 = ConvBNReLU(self.out_channels[9], self.out_channels[9], kernel_size=3, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.out_channels[9], self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inverted_residuals(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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