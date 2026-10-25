import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.layer0 = conv_bn(self.in_channels, 32, stride=2, leaky=0.1)
        self.layer1 = conv_dw(32, 32, stride=1, leaky=0.1)
        self.layer2 = conv_dw(32, 64, stride=2, leaky=0.1)
        self.layer3 = conv_dw(64, 64, stride=1, leaky=0.1)
        self.layer4 = conv_dw(64, 128, stride=2, leaky=0.1)
        self.layer5 = conv_dw(128, 128, stride=1, leaky=0.1)
        self.layer6 = conv_dw(128, 128, stride=1, leaky=0.1)
        self.layer7 = conv_dw(128, 256, stride=2, leaky=0.1)
        self.layer8 = conv_dw(256, 256, stride=1, leaky=0.1)
        self.layer9 = conv_dw(256, 256, stride=1, leaky=0.1)
        self.layer10 = conv_dw(256, 256, stride=1, leaky=0.1)
        self.layer11 = conv_dw(256, 256, stride=1, leaky=0.1)
        self.layer12 = conv_dw(256, 256, stride=1, leaky=0.1)
        self.layer13 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(256, 1024, 1, 1, 0),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()