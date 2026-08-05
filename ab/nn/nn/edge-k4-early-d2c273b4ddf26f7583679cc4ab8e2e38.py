import torch
import torch.nn as nn
from torchvision import transforms

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Conv(nn.Module):

    def __init__(self, inp, oup, k=1, s=1, p=None, g=1, d=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, self._pad(k, p), d, g, False)
        self.norm = nn.BatchNorm2d(oup)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    @staticmethod
    def _pad(k, p):
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        return p

class CSPBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CSPBlock, self).__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(2 * (out_ch // 2), out_ch)
        self.res_m = nn.Sequential(Conv(out_ch // 2, out_ch // 2, k=1), Conv(out_ch // 2, out_ch // 2, k=3, s=1, p=1))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        cat = torch.cat((y, self.conv2(x)), dim=1)
        return self.conv3(cat)

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.feature_extractor = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.dropout = nn.Dropout(prm['dropout'])
        self.classifier = nn.Linear(1024 * 7 * 7, out_shape[0])

    def build_features(self):
        layers = []
        layers.append(Conv(3, 32, k=3, s=1, p=1, act=True))
        layers.append(CSPBlock(32, 32))
        layers.append(Conv(32, 64, k=3, s=2, p=1, act=True))
        layers.append(CSPBlock(64, 64))
        layers.append(Conv(64, 128, k=3, s=2, p=1, act=True))
        layers.append(CSPBlock(128, 128))
        layers.append(Conv(128, 256, k=3, s=2, p=1, act=True))
        layers.append(CSPBlock(256, 256))
        layers.append(Conv(256, 512, k=3, s=2, p=1, act=True))
        layers.append(CSPBlock(512, 512))
        layers.append(Conv(512, 1024, k=3, s=2, p=1, act=True))
        layers.append(CSPBlock(1024, 1024))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=0.0001)

    def learn(self, train_data):
        self.train()
        for inputs, targets in train_data:
            inputs, targets = (inputs.to(self.device), targets.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()