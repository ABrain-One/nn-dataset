import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=False),
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=False),
            nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.stride == 1 and self.expand_ratio == 1:
            return x + self.block(x)
        else:
            return self.block(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.inverted1 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.inverted2 = InvertedResidual(32, 32, stride=2, expand_ratio=3)
        self.inverted3 = InvertedResidual(32, 64, stride=2, expand_ratio=6)
        self.inverted4 = InvertedResidual(64, 64, stride=1, expand_ratio=6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inverted1(x)
        x = self.inverted2(x)
        x = self.inverted3(x)
        x = self.inverted4(x)
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