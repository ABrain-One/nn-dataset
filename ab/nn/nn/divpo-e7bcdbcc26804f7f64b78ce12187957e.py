import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(in_channels * expand_ratio))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, padding=dilation, dilation=dilation, groups=in_channels * expand_ratio, bias=False))
        layers.append(nn.BatchNorm2d(in_channels * expand_ratio))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(EfficientNet, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = 32
        self.num_classes = out_shape[0]
        self.dropout = prm.get('dropout', 0.2)
        self.num_filters = [32, 16, 24, 40, 80, 96]
        self.expand_ratio = [3, 3, 5, 3, 5, 6]
        self.dilation = [1, 2, 2, 2, 2, 2]

        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_filters[-1], self.num_classes)

    def _make_layers(self):
        layers = []
        in_channels = self.in_channels
        layers.append(nn.Conv2d(in_channels, self.num_filters[0], kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(self.num_filters[0]))
        layers.append(nn.ReLU6(inplace=True))

        for i in range(len(self.num_filters) - 1):
            layers.append(InvertedResidual(
                in_channels=self.num_filters[i],
                out_channels=self.num_filters[i+1],
                stride=2,
                expand_ratio=self.expand_ratio[i],
                dilation=self.dilation[i]
            ))
            in_channels = self.num_filters[i+1]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.1),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criteria(self(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.efficientnet = EfficientNet(in_shape, out_shape, prm, device)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

    def forward(self, x):
        x = self.efficientnet(x)
        return x

    def train_setup(self, prm):
        self.efficientnet.train_setup(prm)

    def learn(self, train_data):
        self.efficientnet.learn(train_data)