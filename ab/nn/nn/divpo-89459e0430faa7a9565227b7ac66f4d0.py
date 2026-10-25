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
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, 3, stride,
                      padding=1, groups=in_channels * expansion_factor, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expansion_factor, out_channels, 1, 1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = InvertedResidual(32, 32, 1, 3)
        self.block2 = InvertedResidual(32, 64, 2, 3)
        self.block3 = InvertedResidual(64, 64, 1, 3)
        self.block4 = InvertedResidual(64, 96, 1, 3)
        self.block5 = InvertedResidual(96, 96, 1, 3)
        self.block6 = InvertedResidual(96, 192, 2, 3)

        self.layers = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(192 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def _to(self, device):
        self.to(device)
        self.stem.to(device)
        self.block1.to(device)
        self.block2.to(device)
        self.block3.to(device)
        self.block4.to(device)
        self.block5.to(device)
        self.block6.to(device)
        self.layers.to(device)
        self.avg_pool.to(device)
        self.dropout.to(device)
        self.classifier.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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