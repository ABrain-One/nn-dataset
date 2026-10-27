import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            ConvBNReLU(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1)
        )
        layers.append(
            ConvBNReLU(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride, padding=1)
        )
        layers.append(
            ConvBNReLU(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1)
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBNReLU(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, activation_fn=None, norm_layer=None
    ):
        super(ConvBNReLU, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        activation_fn = activation_fn or nn.ReLU6

        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
        self.add_module('bn', norm_layer(out_channels))
        self.add_module('relu', activation_fn())

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1)
        )

        layers = []
        layers.append(InvertedResidual(32, 16, stride=1, expand_ratio=4))
        layers.append(InvertedResidual(16, 24, stride=2, expand_ratio=4))
        layers.append(InvertedResidual(24, 32, stride=2, expand_ratio=4))
        layers.append(InvertedResidual(32, 64, stride=2, expand_ratio=4))
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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