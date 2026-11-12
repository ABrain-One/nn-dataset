import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.act(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        block_size = 16
        num_blocks = [block_size, block_size * 2, block_size * 4]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, block_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(block_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block1 = Block(block_size)
        self.layer1 = ResBlock(block_size, block_size * 2, stride=2)
        self.layer2 = ResBlock(block_size * 2, block_size * 4, stride=2)
        self.layer3 = nn.Sequential(
            Block(block_size * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(block_size * 4, num_classes),
        )
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.2))
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.1,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
        )

    def learn(self, data_roll):
        self.train()
        for batch in data_roll:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self(x)
            loss = self.criteria(out, y)
            loss.backward()
            self.optimizer.step()