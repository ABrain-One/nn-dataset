import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Helper block for residual connections
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'shortcut'):
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_shape, out_shape, num_classes, dropout_rate=0.0, weight_decay=0.0):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.device = in_shape[1]

        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        layers.append(BasicBlock(64, 64, stride=1))
        for i in range(1, num_blocks[0]):
            layers.append(BasicBlock(64, 64))

        layers.append(BasicBlock(64, 128, stride=2))
        for i in range(1, num_blocks[1]):
            layers.append(BasicBlock(128, 128))

        layers.append(BasicBlock(128, 256, stride=2))
        for i in range(1, num_blocks[2]):
            layers.append(BasicBlock(256, 256))

        layers.append(BasicBlock(256, 512, stride=2))
        for i in range(1, num_blocks[3]):
            layers.append(BasicBlock(512, 512))

        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layers(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        if hasattr(self, 'dropout'):
            out = self.dropout(out)

        out = self.fc(out)
        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__(
            BasicBlock,
            [2, 2, 2, 2],
            in_shape,
            out_shape,
            num_classes=int(out_shape[0]),
            dropout_rate=prm.get('dropout', 0.0),
            weight_decay=prm.get('weight_decay', 0.0)
        )
        self.device = device

    def forward(self, x):
        return super(Net, self).forward(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.1),
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()