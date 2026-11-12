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
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
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
    def __init__(self, block, num_blocks, num_classes, dropout_rate=0.0):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.features(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        num_blocks = [2, 2, 2, 2]
        num_classes = int(out_shape[0])
        dropout_rate = prm.get('dropout', 0.0)
        super(Net, self).__init__(BasicBlock, num_blocks, num_classes, dropout_rate)
        self.device = device
        
        # Adjust initial convolution to match in_shape[1]
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
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
        if 'weight_decay' in prm:
            self.optimizer.param_groups[0]['weight_decay'] = prm.get('weight_decay', 0.0)

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