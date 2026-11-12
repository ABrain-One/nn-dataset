import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layer_channels = in_channels * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, layer_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(layer_channels),
            nn.ReLU6(inplace=True),
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(layer_channels, layer_channels, kernel_size=3, stride=stride, padding=1, groups=layer_channels, bias=False),
            nn.BatchNorm2d(layer_channels),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(layer_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.project_conv(self.depthwise_conv(self.expand_conv(x)))
        else:
            return self.project_conv(self.depthwise_conv(self.expand_conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 64, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(64, 128, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

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
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.criteria[0](out, y)
            loss.backward()
            self.optimizer.step()