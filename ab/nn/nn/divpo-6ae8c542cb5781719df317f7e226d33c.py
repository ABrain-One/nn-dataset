import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'downsample'):
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'downsample'):
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_shape, out_shape, num_classes=10, dropout=0.0):
        super().__init__()
        self.in_channels = 64
        in_channels = in_shape[1]
        self.device = in_shape[0].device if isinstance(in_shape[0], torch.Tensor) else None

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential()
        stride = 2
        self.layers.add_module(f"layer1_0", block(self.in_channels, 64, stride))
        self.in_channels = 64 * block.expansion
        for idx in range(1, layers[0]):
            self.layers.add_module(f"layer1_{idx}", block(self.in_channels, 64))

        stride = 2 if in_channels != 128 * block.expansion else 1
        self.layers.add_module(f"layer2_0", block(self.in_channels, 128, stride))
        self.in_channels = 128 * block.expansion
        for idx in range(1, layers[1]):
            self.layers.add_module(f"layer2_{idx}", block(self.in_channels, 128))

        stride = 2 if in_channels != 256 * block.expansion else 1
        self.layers.add_module(f"layer3_0", block(self.in_channels, 256, stride))
        self.in_channels = 256 * block.expansion
        for idx in range(1, layers[2]):
            self.layers.add_module(f"layer3_{idx}", block(self.in_channels, 256))

        stride = 2 if in_channels != 512 * block.expansion else 1
        self.layers.add_module(f"layer4_0", block(self.in_channels, 512, stride))
        self.in_channels = 512 * block.expansion
        for idx in range(1, layers[3]):
            self.layers.add_module(f"layer4_{idx}", block(self.in_channels, 512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        x = self.fc(x)

        return x

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        num_classes = int(out_shape[0])
        in_channels = in_shape[1]
        blocks = [2, 2, 2, 2]  # For ResNet18
        super().__init__(BasicBlock, blocks, in_shape, out_shape, num_classes=num_classes, dropout=prm.get('dropout', 0.0))
        self.device = device

    def forward(self, x):
        return super().forward(x)

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