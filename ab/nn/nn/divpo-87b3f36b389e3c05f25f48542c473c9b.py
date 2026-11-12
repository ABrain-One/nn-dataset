import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3_large(nn.Module):
    def __init__(self, in_shape=(3, 32, 32), out_shape=(10,), prm=None, device=None):
        super(MobileNetV3_large, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.block_1 = InvertedResidual(16, 16, 1, 1)
        self.block_2 = InvertedResidual(16, 24, 2, 6)
        self.block_3 = InvertedResidual(24, 24, 1, 6)
        self.block_4 = InvertedResidual(24, 32, 2, 6)
        self.block_5 = InvertedResidual(32, 32, 1, 6)
        self.block_6 = InvertedResidual(32, 64, 2, 6)
        self.block_7 = InvertedResidual(64, 64, 1, 6)
        self.block_8 = InvertedResidual(64, 96, 1, 6)
        self.block_9 = InvertedResidual(96, 96, 2, 6)
        self.block_10 = InvertedResidual(96, 160, 1, 6)
        self.block_11 = InvertedResidual(160, 160, 2, 6)
        self.block_12 = InvertedResidual(160, 320, 1, 6)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, num_classes)

        self.to(self.device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.block_9(x)
        x = self.block_10(x)
        x = self.block_11(x)
        x = self.block_12(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.mobilenet = MobileNetV3_large(in_shape, out_shape, prm, device)
        self.conv = nn.Conv2d(in_shape[1], 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return self.mobilenet(x)

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