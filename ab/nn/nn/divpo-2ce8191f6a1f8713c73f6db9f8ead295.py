import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class ConvBNAct(nn.Sequential):
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1, act_layer=nn.ReLU):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_chs),
            act_layer(inplace=True)
        )

class DepthwiseConv(nn.Sequential):
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0, act_layer=nn.ReLU):
        super(DepthwiseConv, self).__init__(
            nn.Conv2d(in_chs, in_chs, kernel_size, stride, padding, groups=in_chs, bias=False),
            nn.BatchNorm2d(in_chs),
            act_layer(inplace=True),
            nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chs)
        )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = ConvBNAct(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.layer1 = DepthwiseConv(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = DepthwiseConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer3 = DepthwiseConv(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = DepthwiseConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer5 = DepthwiseConv(128, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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