import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += self.downsample(identity)
        out = self.relu(out)
        
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout_p=0.0):
        super(CIFARResidualNet, self).__init__()
        
        # Stem: 3x3 Conv2d, norm, activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: residual blocks at 32x32
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, dropout_p=dropout_p),
            ResidualBlock(64, 64, stride=1, dropout_p=dropout_p)
        )
        
        # Stage 2: residual blocks at 16x16
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout_p=dropout_p),
            ResidualBlock(128, 128, stride=1, dropout_p=dropout_p)
        )
        
        # Stage 3: residual blocks at 8x8
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout_p=dropout_p),
            ResidualBlock(256, 256, stride=1, dropout_p=dropout_p)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.class_number = out_shape[0]
        
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        # Use a small dropout value to satisfy the "optional Dropout2d" part of the description
        # while keeping it minimal for a 1-epoch training scenario.
        self.model = CIFARResidualNet(self.channel_number, self.class_number, dropout_p=0.1)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()