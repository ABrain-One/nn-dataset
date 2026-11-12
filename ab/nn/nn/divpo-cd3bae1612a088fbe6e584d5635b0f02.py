import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        x = self.block(x)
        return nn.ReLU(inplace=True)(x + identity)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64, stride=2),
            Block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, stride=2),
            Block(128, 128),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None
        self._block = Block

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.criteria[0](out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm25d(in_channels)
        self.conv_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm25d(in_channels)

    def forward(self, x):
        residual = self.conv_1(x)
        residual = self.bn_1(residual)
        residual = self.relu_1(residual)
        residual = self.conv_2(residual)
        residual = self.bn_2(residual)
        residual = self.relu_1(residual)
        residual = self.conv_3(residual)
        residual = self.bn_3(residual)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_1(x)
        x = self.conv_3(x)
        x = self.bn_3(x)

        x = x + residual
        x = self.relu_1(x)
        return x