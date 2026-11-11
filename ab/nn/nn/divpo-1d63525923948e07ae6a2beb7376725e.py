import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        layers = []
        self.expanded_channels = int(input_channels * expand_ratio)
        if expand_ratio != 1:
            layers.append(
                Conv2d(input_channels, self.expanded_channels, kernel_size=1, padding=0)
            )

        layers.extend([
            Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=3, padding=1, stride=self.stride, groups=self.expanded_channels),
            nn.Conv2d(self.expanded_channels, output_channels, kernel_size=1, padding=0, stride=1, bias=False)
        ])

        self.conv = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.bn(self.conv(x))
        else:
            return self.bn(self.conv(x))

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=False, groups=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.channels = 128
        self.image_size = 16
        self.blocks = nn.Sequential(
            InvertedResidual(self.channels, self.channels, 1, 2),
            InvertedResidual(self.channels, self.channels, 2, 3),
            InvertedResidual(self.channels, self.channels, 1, 3),
            InvertedResidual(self.channels, self.channels, 1, 4),
            InvertedResidual(self.channels, self.channels, 2, 4),
            InvertedResidual(self.channels, self.channels, 1, 4),
            InvertedResidual(self.channels, self.channels, 1, 4),
            InvertedResidual(self.channels, self.channels, 2, 4),
            InvertedResidual(self.channels, self.channels, 1, 4),
            InvertedResidual(self.channels, self.channels, 1, 4),
        )
        self.conv2d = Conv2d(self.channels, 128, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, int(out_shape[0]))

        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = self.blocks(x)
        x = self.conv2d(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()