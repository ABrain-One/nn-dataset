import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class Head(nn.Module):

    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        p = float(prm['dropout'])
        self.projector = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False)
        self.se = SEBlock(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.projector(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))

class SEBlock(nn.Module):

    def __init__(self, channel):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, 128, bias=False), nn.ReLU(inplace=True), nn.Linear(128, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(residual + out)

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.dropout = float(prm['dropout'])
        self.batch = int(prm['batch'])
        self.learning_rate = float(prm['lr'])
        self.momentum_value = float(prm['momentum'])
        self.num_epochs = int(prm['epoch'])
        self._transform_name = prm['transform']
        in_channels = in_shape[1]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(ResBlock(64, 256, stride=1))
        self.layer2 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512, stride=1))
        self.layer3 = nn.Sequential(ResBlock(512, 1024, stride=2), ResBlock(1024, 1024, stride=1))
        self.layer4 = nn.Sequential(ResBlock(1024, 1024, stride=2), ResBlock(1024, 1024, stride=1))
        self.head = Head(1024, out_shape[0], prm)
        self.to(device)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x)
        return self.head(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm['lr']))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            loss = self.criteria(self(inputs), labels)
            loss.backward()
            self.optimizer.step()