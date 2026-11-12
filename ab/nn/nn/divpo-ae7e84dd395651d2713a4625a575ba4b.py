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
        self.conv1 = nn.Conv2d(self.in_channels, 64, 3, padding=1)
        self.part1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.part2 = SmallBlock(64, 64)
        self.part3 = SmallBlock(64, 64)
        self._avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self._classification = nn.Linear(64, out_shape[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self._avg_pooling(x)
        x = x.flatten(1)
        x = self._classification(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm['lr'], momentum=prm['momentum'],
            weight_decay=1e-4
        )

    def learn(self, train_data):
        self.train()
        for data, trgt in train_data:
            data, trgt = data.to(self.device), trgt.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, trgt)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()