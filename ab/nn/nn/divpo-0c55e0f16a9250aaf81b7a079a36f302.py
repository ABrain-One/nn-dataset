import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        red_dim = in_channels * expansion_factor
        self.conv = nn.Sequential(
            # Pointwise convolution
            nn.Conv2d(in_channels, red_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(red_dim),
            nn.ReLU6(inplace=True),
            # Depthwise convolution
            nn.Conv2d(red_dim, red_dim, 3, stride, 1, groups=red_dim, bias=False),
            nn.BatchNorm2d(red_dim),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(red_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.block = InvertedResidual(in_channels, out_channels, stride, expansion_factor=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = Block(32, 16, stride=1)
        self.block2 = Block(16, 24, stride=2)
        self.block3 = Block(24, 32, stride=2)
        self.block4 = Block(32, 64, stride=2)

        self.expand = nn.Sequential(
            Block(64, 96, stride=1),
            Block(96, 96, stride=1),
            Block(96, 96, stride=1),
            Block(96, 96, stride=1),
        )

        self.conv = nn.Conv2d(96, 16, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.expand(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
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
            self.optimizer.step()