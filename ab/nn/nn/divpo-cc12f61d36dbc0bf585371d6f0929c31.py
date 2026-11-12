import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, dropout=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        layers.append(BasicBlock(self.in_planes, 64, stride=1))
        self.in_planes = 64 * block.expansion
        layers += [BasicBlock(self.in_planes, 64, stride=1) for _ in range(num_blocks[0])]
        
        layers.append(BasicBlock(self.in_planes, 128, stride=2))
        self.in_planes = 128 * block.expansion
        layers += [BasicBlock(self.in_planes, 128, stride=1) for _ in range(num_blocks[1])]
        
        layers.append(BasicBlock(self.in_planes, 256, stride=2))
        self.in_planes = 256 * block.expansion
        layers += [BasicBlock(self.in_planes, 256, stride=1) for _ in range(num_blocks[2])]
        
        layers.append(BasicBlock(self.in_planes, 512, stride=2))
        self.in_planes = 512 * block.expansion
        layers += [BasicBlock(self.in_planes, 512, stride=1) for _ in range(num_blocks[3])]

        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout=prm.get('dropout', 0.0))

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()