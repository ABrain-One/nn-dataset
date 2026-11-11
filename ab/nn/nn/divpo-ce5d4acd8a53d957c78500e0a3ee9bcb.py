import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_shape, out_shape, num_classes, dropout=0.0):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.device = in_shape[1]

        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = []

        layers.append(block(self.in_channels, 64, stride=1))
        self.in_channels = 64 * block.expansion
        for _ in range(1, num_blocks[0]):
            layers.append(block(self.in_channels, 64))

        layers.append(block(self.in_channels, 128, stride=2))
        self.in_channels = 128 * block.expansion
        for _ in range(1, num_blocks[1]):
            layers.append(block(self.in_channels, 128))

        layers.append(block(self.in_channels, 256, stride=2))
        self.in_channels = 256 * block.expansion
        for _ in range(1, num_blocks[2]):
            layers.append(block(self.in_channels, 256))

        layers.append(block(self.in_channels, 512, stride=2))
        self.in_channels = 512 * block.expansion
        for _ in range(1, num_blocks[3]):
            layers.append(block(self.in_channels, 512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        out = self.fc(out)
        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        num_classes = int(out_shape[0])
        in_channels = in_shape[1]
        num_blocks = [2, 2, 2, 2]  # Example configuration for ResNet-18
        super(Net, self).__init__(Bottleneck, num_blocks, in_shape, out_shape, num_classes, dropout=prm.get('dropout', 0.0))
        self.device = device

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0)
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