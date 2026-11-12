import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class FastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return out

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.height = in_shape[2]
        self.width = in_shape[3]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.in_channels * 8
        self.conv1 = nn.Conv2d(self.in_channels, self.features, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.features)
        self.nested = nn.ModuleList([])
        self.last_features = self.features
        self.build_modules()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.last_features, self.num_classes)

    def build_modules(self):
        stride = 1
        features = self.features
        for idx in range(2):
            reduction = (idx == 1)
            block = FastBlock
            add_channel = features // 2 if reduction else 0
            for _ in range(2):
                self.nested.append(block(self.last_features, features, stride))
                self.last_features = features
                self.features += add_channel

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nested[0](x)
        for layer in self.nested[1:]:
            x = layer(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.linear(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_loader):
        self.train()
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()