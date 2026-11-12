import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Helper block
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

# Main Network Class
class ResNet(nn.Module):
    def __init__(self, block, layers, in_shape, out_shape, num_classes=10, dropout_rate=0.0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_in_features = 512 * block.expansion
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.fc_in_features, num_classes)

        self.apply(self.weight_init)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias.data, 0)
            nn.init.constant_(m.weight.data, 1)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.01)
            nn.init.constant_(m.bias.data, 0)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Derived Classes for specific ResNet architectures
class ResNet18(ResNet):
    def __init__(self, in_shape, out_shape, num_classes=10, dropout_rate=0.0):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], in_shape, out_shape, num_classes=num_classes, dropout_rate=dropout_rate)

class ResNet34(ResNet):
    def __init__(self, in_shape, out_shape, num_classes=10, dropout_rate=0.0):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], in_shape, out_shape, num_classes=num_classes, dropout_rate=dropout_rate)

class ResNet50(nn.Module):
    def __init__(self, in_shape, out_shape, num_classes=10, dropout_rate=0.0):
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super(ResNet50, self).__init__(block, layers, in_shape, out_shape, num_classes=num_classes, dropout_rate=dropout_rate)

# Ensure proper inheritance for ResNet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet101(ResNet):
    def __init__(self, in_shape, out_shape, num_classes=10, dropout_rate=0.0):
        block = Bottleneck
        layers = [3, 4, 23, 3]
        super(ResNet101, self).__init__(block, layers, in_shape, out_shape, num_classes=num_classes, dropout_rate=dropout_rate)

class ResNet152(ResNet):
    def __init__(self, in_shape, out_shape, num_classes=10, dropout_rate=0.0):
        block = Bottleneck
        layers = [3, 8, 36, 3]
        super(ResNet152, self).__init__(block, layers, in_shape, out_shape, num_classes=num_classes, dropout_rate=dropout_rate)

# Net class definition
class Net(ResNet18):
    def __init__(self, in_shape, out_shape, prm, device):
        num_classes = int(out_shape[0])
        dropout_rate = prm.get('dropout', 0.0)
        super(Net, self).__init__(in_shape=in_shape, out_shape=out_shape, num_classes=num_classes, dropout_rate=dropout_rate)
        self.device = device
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        if 'weight_decay' in prm:
            self.optimizer.param_groups[0]['weight_decay'] = prm['weight_decay']

    def learn(self, train_data):
        self.train()
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[Epoch {i // 100 + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0