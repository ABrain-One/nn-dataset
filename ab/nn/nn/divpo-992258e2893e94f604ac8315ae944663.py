import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DartsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DartsBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_path = DropPath(drop_prob=0.1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out = out + self.shortcut(x)
        return torch.relu(out)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per item/batch, compute drop mask
        drop_mask = torch.empty([x.size()[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        drop_mask = drop_mask.bernoulli_(keep_prob)
        # expand mask
        output = x.div(keep_prob) * drop_mask
    else:
        output = x
    return output

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super(Net, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        self._channels = 32
        self._features = 32
        self.conv1 = nn.Conv2d(in_shape[1], self._channels, kernel_size=3, padding=1, stride=1)
        self.nam = nn.Sequential().eval()
        self.layers = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classification = nn.Linear(32 * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.nam(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classification(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'],
                                          weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            self.optimizer.zero_grad()
            output = self.forward(data.to(self.device))
            loss = self.criteria(output, target.to(self.device))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()