import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNAct, self).__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class Block(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, groups=1, apply_act=True):
        super(Block, self).__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=3, stride=stride, groups=groups)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=3, stride=1, groups=groups)
        self.apply_act = apply_act

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = ConvBNAct(in_shape[1], 32, kernel_size=3, stride=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.criteria = nn.CrossEntropyLoss().to(self.device)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()