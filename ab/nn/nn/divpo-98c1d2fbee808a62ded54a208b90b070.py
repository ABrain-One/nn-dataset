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

def conv_bn_no_relu(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_no_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class ResBlock(nn.Module):
    def __init__(self, channel_num, leaky=0):
        super(ResBlock, self).__init__()
        self.conv1 = conv_bn_no_relu(channel_num, channel_num, stride=1, leaky=leaky)
        self.conv2 = conv_bn(channel_num, channel_num, stride=1, leaky=leaky)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.layer0 = conv_bn(self.in_channels, 32, stride=2, leaky=0.1)
        self.layer1 = conv_bn(32, 32, stride=1, leaky=0.1)
        self.layer2 = conv_bn(32, 64, stride=2, leaky=0.1)
        self.layer3 = conv_bn(64, 64, stride=1, leaky=0.1)
        self.layer4 = ResBlock(64, leaky=0.1)
        self.layer5 = conv_bn(64, 128, stride=2, leaky=0.1)
        self.layer6 = conv_bn(128, 128, stride=1, leaky=0.1)
        self.layer7 = conv_bn(128, 128, stride=1, leaky=0.1)
        self.layer8 = ResBlock(128, leaky=0.1)
        self.layer9 = conv_bn(128, 128, stride=1, leaky=0.1)
        self.layer10 = conv_bn_no_relu(128, 128, stride=2, leaky=0.1)
        self.layer11 = conv_bn(128, 256, stride=2, leaky=0.1)
        self.layer12 = conv_bn(256, 256, stride=1, leaky=0.1)
        self.layer13 = conv_bn(256, 256, stride=1, leaky=0.1)
        self.layer14 = conv_bn(256, 256, stride=1, leaky=0.1)
        self.layer15 = conv_bn(256, 256, stride=1, leaky=0.1)
        self.layer16 = conv_bn_no_relu(256, 256, stride=2, leaky=0.1)
        self.conv_last = conv_no_bn(256, self.num_classes, stride=1, leaky=0.1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

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
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.conv_last(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.softmax(x)

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