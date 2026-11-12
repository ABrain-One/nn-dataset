import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Helper block for residual blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Residual Network Model
class ResNet(nn.Module):
    def __init__(self, block, layers, in_shape, out_shape, num_classes, dropout=0.0, weight_decay=0.0, device='cpu'):
        super(ResNet, self).__init__()
        self.device = device
        in_channels = in_shape[1]

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential()
        strides = [2, 1, 1, 1]
        in_channels = 64
        for i in range(len(layers)):
            self.layers.add_module(f"layer{i+1}", self._make_layer(block, out_channels=64 * block.expansion, blocks=layers[i], stride=strides[i]))
            in_channels = 64 * block.expansion

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(lambda m: setattr(m, 'weight_decay', weight_decay) if hasattr(m, 'weight_decay') else None)
        self.weight_decay = weight_decay

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layers(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc(out)

        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device='cpu'):
        num_classes = int(out_shape[0])
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        dropout = prm.get('dropout', 0.0)
        weight_decay = prm.get('weight_decay', 0.0)
        
        super(Net, self).__init__(
            BasicBlock, 
            [2, 2, 2, 2], 
            in_shape, 
            out_shape, 
            num_classes, 
            dropout=dropout, 
            weight_decay=weight_decay, 
            device=device
        )

    def forward(self, x):
        return super(Net, self).forward(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9), 
            weight_decay=self.weight_decay
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