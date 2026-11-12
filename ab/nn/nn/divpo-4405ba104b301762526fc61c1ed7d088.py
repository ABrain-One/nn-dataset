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
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Instantiating the network for CIFAR-10
def resnet18(in_shape, out_shape, num_classes=10, dropout_rate=0.0):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_shape, out_shape, num_classes=num_classes, dropout_rate=dropout_rate)
    return model

# Wrapper class for easy initialization and training setup
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.model = resnet18(in_shape, out_shape, num_classes=int(out_shape[0]), 
                             dropout_rate=prm.get('dropout', 0.0)).to(device)
        self.in_channels = in_shape[1]
        self.num_classes = int(out_shape[0])

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                          lr=prm.get('lr', 0.01), 
                                          momentum=prm.get('momentum', 0.9), 
                                          weight_decay=prm.get('weight_decay', 0.0))

    def learn(self, train_data):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_data)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

# Example usage:
# net = Net((3, 32, 32), (10,), {'lr': 0.01, 'momentum': 0.9}, 'cuda')
# net.train_setup({'lr': 0.01, 'momentum': 0.9})
# epoch_loss, epoch_acc = net.learn(train_loader)