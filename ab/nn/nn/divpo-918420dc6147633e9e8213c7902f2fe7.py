import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(leaky)
    )

def conv_dw(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(leaky),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(leaky)
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = conv_bn(in_shape[1], 32, stride=2, leaky=0.1)
        self.conv2 = conv_bn(32, 64, stride=2, leaky=0.1)
        self.conv3 = conv_dw(64, 128, stride=2, leaky=0.1)
        self.conv4 = conv_dw(128, 128, stride=1, leaky=0.1)
        self.conv5 = conv_dw(128, 256, stride=2, leaky=0.1)
        self.conv6 = conv_dw(256, 256, stride=1, leaky=0.1)
        self.conv7 = conv_dw(256, 512, stride=2, leaky=0.1)
        self.conv8 = conv_dw(512, 512, stride=1, leaky=0.1)
        self.conv9 = conv_dw(512, 512, stride=1, leaky=0.1)
        self.conv10 = conv_dw(512, 512, stride=1, leaky=0.1)
        self.conv11 = conv_dw(512, 512, stride=1, leaky=0.1)

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, out_shape[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = self.avg(x)
        x = x.flatten(1)
        x = self.fc(x)
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}