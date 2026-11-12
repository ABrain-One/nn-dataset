import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.downsample(orig)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        this_in = self.in_channels
        self.features = nn.Conv2d(self.in_channels, 32, 3, padding=1)
        self.classic_block1 = SmallBlock(32, 32)
        self.classic_block2 = SmallBlock(32, 32)
        self.this_in = 32
        self.this_in2 = 32
        self.features2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.classic_block3 = SmallBlock(32, 64)
        self.classic_block4 = SmallBlock(64, 64)
        self.this_in3 = 64
        self.this_in4 = 64
        self.features3 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.classic_block5 = SmallBlock(64, 128)
        self.classic_block6 = SmallBlock(128, 128)
        selfthis_in5 = 128
        self.this_in6 = 128
        self.x = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.classic_block7 = SmallBlock(128, 128)
        self.classic_block8 = SmallBlock(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.class_2 = nn.Sequential(nn.Linear(128 * 7 * 7, 100))
        self._last_x = self._make_layer(nn.Sequential(nn.Linear(128 * 7 * 7, 100)), 100, 10)
        self.classifier = nn.Linear(100, out_shape[0])

    def _make_layer(self, x, out_c, layers):
        return x

    def forward(self, x):
        x = x.to(self.device)
        x = self.features(x)
        x = self.classic_block1(x)
        x = self.classic_block2(x)
        x = self.features2(x)
        x = self.classic_block3(x)
        x = self.classic_block4(x)
        x = self.features3(x)
        x = self.classic_block5(x)
        x = self.classic_block6(x)
        x = self.x(x)
        x = self.classic_block7(x)
        x = self.classic_block8(x)
        x = nn.functional.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), 128, 7, 7)
        x = x.view(x.size(0), 128 * 7 * 7)
        x = self._last_x(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm['lr'], momentum=prm['momentum']
        )

    def learn(self, data):
        self.train()
        for inps, tgs in data:
            inps, tgs = inps.to(self.device), tgs.to(self.device)
            self.optimizer.zero_grad()
            outs = self(inps)
            ls = self.criterion(outs, tgs)
            ls.backward()
            self.optimizer.step()