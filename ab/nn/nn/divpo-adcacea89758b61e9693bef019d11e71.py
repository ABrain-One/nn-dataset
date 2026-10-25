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
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride, expansion_factor=3),
            nn.ReLU(inplace=True),
            InvertedResidual(out_channels, out_channels, 1, expansion_factor=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = Block(32, 16, stride=1)
        self.block2 = Block(16, 24, stride=2)
        self.block3 = Block(24, 32, stride=2)
        self.block4 = Block(32, 64, stride=2)
        self.block5 = Block(64, 96, stride=1)

        self.features = nn.Sequential(
            self.stem,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(96, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
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