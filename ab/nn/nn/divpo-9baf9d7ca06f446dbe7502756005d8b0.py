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

def bottleneck_block(in_channels, out_channels, stride, expand_ratio):
    hidden_channels = int(out_channels * expand_ratio)
    return nn.Sequential(
        conv_bn(in_channels, hidden_channels, kernel_size=1, stride=1),
        conv_bn(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1),
        conv_bn(hidden_channels, out_channels, kernel_size=1, stride=1)
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = conv_bn(self.in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bottleneck1 = bottleneck_block(32, 32, stride=1, expand_ratio=1)
        self.bottleneck2 = bottleneck_block(32, 32, stride=1, expand_ratio=4)
        self.bottleneck3 = bottleneck_block(32, 32, stride=1, expand_ratio=4)
        self.conv2 = conv_bn(32, 64, kernel_size=3, stride=2, padding=1)
        self.bottleneck4 = bottleneck_block(64, 64, stride=1, expand_ratio=4)
        self.bottleneck5 = bottleneck_block(64, 64, stride=1, expand_ratio=4)
        self.bottleneck6 = bottleneck_block(64, 64, stride=1, expand_ratio=4)
        self.conv3 = conv_bn(64, 128, kernel_size=3, stride=2, padding=1)
        self.bottleneck7 = bottleneck_block(128, 128, stride=1, expand_ratio=4)
        self.bottleneck8 = bottleneck_block(128, 128, stride=1, expand_ratio=4)
        self.bottleneck9 = bottleneck_block(128, 128, stride=1, expand_ratio=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.conv2(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.conv3(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()