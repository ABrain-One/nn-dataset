import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),
        ]

        layers += [
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),
        ]

        layers += [
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block1 = InvertedResidual(32, 32, stride=1, expansion_factor=3)
        self.block2 = InvertedResidual(32, 32, stride=1, expansion_factor=3)
        self.block3 = InvertedResidual(32, 64, stride=2, expansion_factor=3)
        self.block4 = InvertedResidual(64, 64, stride=1, expansion_factor=3)
        self.block5 = InvertedResidual(64, 96, stride=1, expansion_factor=3)
        self.block6 = InvertedResidual(96, 96, stride=1, expansion_factor=3)
        self.block7 = InvertedResidual(96, 160, stride=2, expansion_factor=3)
        self.block8 = InvertedResidual(160, 160, stride=1, expansion_factor=3)
        self.block9 = InvertedResidual(160, 320, stride=1, expansion_factor=3)

        self.conv = nn.Conv2d(320, 160, kernel_size=1, stride=1, padding=0)
        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(160, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.conv(x)
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
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
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()