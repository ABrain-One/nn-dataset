import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper block
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block = Block(32, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(32 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

# Helper Block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.Sequential(
            ResBlock(32, 32, stride=1, expansion=3),
            ResBlock(32, 32, stride=2, expansion=3),
            ResBlock(32, 64, stride=2, expansion=3),
            ResBlock(64, 64, stride=1, expansion=3),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.dropout(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(ResBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu1(out + residual)