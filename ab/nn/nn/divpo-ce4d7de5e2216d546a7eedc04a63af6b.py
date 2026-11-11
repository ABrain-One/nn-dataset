import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = []

        layers.append(BasicBlock(self.in_channels, 64, stride=1))
        self.in_channels = 64 * block.expansion
        layers += [BasicBlock(self.in_channels, 64) for _ in range(1, num_blocks[0])]

        layers.append(BasicBlock(self.in_channels, 128, stride=2))
        self.in_channels = 128 * block.expansion
        layers += [BasicBlock(self.in_channels, 128) for _ in range(1, num_blocks[1])]

        layers.append(BasicBlock(self.in_channels, 256, stride=2))
        self.in_channels = 256 * block.expansion
        layers += [BasicBlock(self.in_channels, 256) for _ in range(1, num_blocks[2])]

        layers.append(BasicBlock(self.in_channels, 512, stride=2))
        self.in_channels = 512 * block.expansion
        layers += [BasicBlock(self.in_channels, 512) for _ in range(1, num_blocks[3])]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        num_blocks = [2, 2, 2, 2]
        num_classes = int(out_shape[0])
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        super(Net, self).__init__(BasicBlock, num_blocks, num_classes)
        self.device = device
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ] + [
            self._make_layer(64, 64, blocks=num_blocks[0]),
            self._make_layer(64, 128, stride=2, blocks=num_blocks[1]),
            self._make_layer(128, 256, stride=2, blocks=num_blocks[2]),
            self._make_layer(256, 512, stride=2, blocks=num_blocks[3])
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        p = prm.get('dropout', 0.0)
        if p > 0:
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = None
        self.apply(self._initialize_weights)

    def _make_layer(self, in_channels, out_channels, stride=1, blocks=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc(out)
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        p = prm.get('dropout', 0.0)
        if p > 0:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p

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