import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
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
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_shape, dropout=0.0, weight_decay=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dropout = dropout
        
        self.device = in_shape[0]
        self.block = block
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=dropout)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        if self.weight_decay is not None:
            self.apply(lambda m: setattr(m, "weight_decay", self.weight_decay))

    def _make_layer(self, block, planes, blocks, stride, dropout):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride, dropout))
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        num_classes = int(out_shape[0])
        num_blocks = [2, 2, 2, 2]  # Example configuration, can be adjusted
        
        self.resnet = ResNet(BasicBlock, num_blocks, num_classes, in_shape, dropout=prm.get('dropout', 0.0), weight_decay=prm.get('weight_decay', 0.0))
        
    def forward(self, x):
        return self.resnet(x)
    
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