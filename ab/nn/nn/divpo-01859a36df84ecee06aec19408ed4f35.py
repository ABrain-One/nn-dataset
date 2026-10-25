import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion = expansion

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels * self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.expansion, self.in_channels * self.expansion, kernel_size=3, stride=self.stride, padding=1, groups=self.in_channels * self.expansion, bias=False),
            nn.BatchNorm2d(self.in_channels * self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.expansion, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.bottleneck(x) + self.shortcut(x))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 32, stride=2)
        self.layer3 = Block(32, 32, stride=1)
        self.layer4 = Block(32, 32, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
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
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()