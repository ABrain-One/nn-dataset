import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.conv1 = conv_bn(self.in_channels, 32, 2)
        self.inv1 = InvertedResidual(32, 32, 1, 1)
        self.inv2 = InvertedResidual(32, 32, 1, 3)
        self.inv3 = InvertedResidual(32, 64, 2, 3)
        self.inv4 = InvertedResidual(64, 64, 1, 3)
        self.inv5 = InvertedResidual(64, 128, 2, 3)
        self.inv6 = InvertedResidual(128, 128, 1, 3)
        self.conv2 = conv_1x1_bn(128, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self._parameters = {
            'conv1.0.weight': torch.randn(32, self.in_channels, 3, 3),
            'inv1.conv.0.weight': torch.randn(32, 32, 1, 1),
            'inv1.conv.1.weight': torch.randn(32, 32, 3, 3),
            'inv1.conv.3.weight': torch.randn(32, 32, 1, 1),
            'inv2.conv.0.weight': torch.randn(32, 32, 1, 1),
            'inv2.conv.1.weight': torch.randn(32, 32, 3, 3),
            'inv2.conv.3.weight': torch.randn(32, 32, 1, 1),
            'inv3.conv.0.weight': torch.randn(32, 32, 1, 1),
            'inv3.conv.1.weight': torch.randn(32, 32, 3, 3),
            'inv3.conv.3.weight': torch.randn(32, 32, 1, 1),
            'inv4.conv.0.weight': torch.randn(32, 32, 1, 1),
            'inv4.conv.1.weight': torch.randn(32, 32, 3, 3),
            'inv4.conv.3.weight': torch.randn(32, 32, 1, 1),
            'inv5.conv.0.weight': torch.randn(32, 32, 1, 1),
            'inv5.conv.1.weight': torch.randn(64, 32, 3, 3),
            'inv5.conv.3.weight': torch.randn(64, 64, 1, 1),
            'inv6.conv.0.weight': torch.randn(64, 64, 1, 1),
            'inv6.conv.1.weight': torch.randn(128, 64, 3, 3),
            'inv6.conv.3.weight': torch.randn(128, 128, 1, 1),
            'conv2.0.weight': torch.randn(128, 32, 1, 1),
            'fc.weight': torch.randn(self.num_classes, 128),
        }

        self._parameters_v2 = self._parameters.copy()
        for key in self._parameters_v2.keys():
            if 'weight' in key:
                self._parameters_v2[key] = self._parameters_v2[key].half()

    def forward(self, x):
        x = self.conv1(x)
        x = self.inv1(x)
        x = self.inv2(x)
        x = self.inv3(x)
        x = self.inv4(x)
        x = self.inv5(x)
        x = self.inv6(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}