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

        hidden_dim = int(inp * expand_ratio)
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
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
        self.conv1 = conv_bn(3, 32, 2)
        self.conv2 = conv_bn(32, 32, 1)
        self.conv3 = conv_bn(32, 64, 2)
        self.conv4 = conv_bn(64, 64, 1)
        self.att1 = InvertedResidual(64, 64, 1, 1)
        self.conv5 = conv_bn(64, 128, 2)
        self.conv6 = conv_bn(128, 128, 1)
        self.conv7 = conv_bn(128, 128, 1)
        self.att2 = InvertedResidual(128, 128, 1, 1)
        self.conv8 = conv_bn(128, 256, 2)
        self.conv9 = conv_bn(256, 256, 1)
        self.conv10 = conv_bn(256, 256, 1)
        self.att3 = InvertedResidual(256, 256, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = conv_1x1_bn(256, 128)
        self.last_conv1 = nn.Conv2d(128, 40, 1, 1, 0)
        self.last_bn = nn.BatchNorm2d(40)
        self.last_relu = nn.ReLU(inplace=True)
        self.last_conv2 = nn.Conv2d(40, 10, 1, 1, 0)
        self.num_classes = out_shape[0]

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.att1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.att2(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.att3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.last_conv1(x)
        x = self.last_bn(x)
        x = self.last_relu(x)
        x = self.last_conv2(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()