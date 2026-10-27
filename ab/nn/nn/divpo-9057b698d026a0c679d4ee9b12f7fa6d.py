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
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels * expand_ratio,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels * expand_ratio),
                nn.ReLU6(inplace=True),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels * expand_ratio,
                    in_channels * expand_ratio,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels * expand_ratio,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels * expand_ratio),
                nn.ReLU6(inplace=True),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels * expand_ratio,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

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

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.Sequential(
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            InvertedResidual(32, 64, 2, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            InvertedResidual(64, 96, 2, 2),
            InvertedResidual(96, 96, 1, 2),
            InvertedResidual(96, 96, 1, 2),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(96, 160, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(160, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()