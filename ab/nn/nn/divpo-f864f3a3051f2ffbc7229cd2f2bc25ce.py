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

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup)
    )

def conv_no_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv_bn(in_channels, out_channels, stride)
        self.conv2 = conv_bn(out_channels, out_channels, 1)
        self.downsample = conv_bn_no_relu(in_channels, out_channels, stride)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if residual.shape != out.shape:
            residual = self.downsample(x)
        out = out + residual
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = conv_bn(self.in_channels, 32, 2)
        self.res1 = ResBlock(32, 32, 1)
        self.res2 = ResBlock(32, 64, 2)
        self.res3 = ResBlock(64, 128, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.softmax(self.fc(x))

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