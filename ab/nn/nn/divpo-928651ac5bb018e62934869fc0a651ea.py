import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, kernel_size, stride, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def bottleneck_block(in_channels, mid_channels, out_channels, stride, expand_ratio):
    hidden_channels = mid_channels * expand_ratio
    return nn.Sequential(
        conv_bn(in_channels, hidden_channels, 1, 1),
        conv_bn(hidden_channels, hidden_channels, 3, stride, padding=1),
        conv_bn(hidden_channels, out_channels, 1, 1)
    )

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
        self.block1 = bottleneck_block(32, 16, 32, 1, 1)
        self.block2 = bottleneck_block(32, 16, 64, 2, 6)
        self.block3 = bottleneck_block(64, 24, 72, 2, 6)
        self.block4 = bottleneck_block(72, 24, 72, 1, 6)
        self.block5 = bottleneck_block(72, 40, 96, 2, 6)
        self.block6 = bottleneck_block(96, 40, 96, 1, 6)
        self.block7 = bottleneck_block(96, 40, 96, 1, 6)
        self.block8 = bottleneck_block(96, 48, 144, 2, 6)
        self.block9 = bottleneck_block(144, 48, 144, 1, 6)
        self.block10 = bottleneck_block(144, 48, 144, 1, 6)
        self.block11 = bottleneck_block(144, 64, 192, 2, 6)
        self.block12 = bottleneck_block(192, 64, 192, 1, 6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, num_classes)

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
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()