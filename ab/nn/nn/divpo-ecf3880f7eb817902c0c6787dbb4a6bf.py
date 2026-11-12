import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Helper block for residual connections
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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

# Residual Network Definition
class ResNet(nn.Module):
    def __init__(self, block, layers, in_shape, out_shape, num_classes, dropout=0.0):
        super(ResNet, self).__init__()
        in_channels = in_shape[1]
        self.device = in_shape[0].device if isinstance(in_shape[0], torch.Tensor) else None
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential()
        in_channels = 64
        stride = 2 if in_channels != 64 else 1
        self.layers.add_module('layer1', self._make_layer(block, 64, layers[0], stride))
        in_channels = 128
        stride = 2
        self.layers.add_module('layer2', self._make_layer(block, 128, layers[1], stride))
        in_channels = 256
        stride = 2
        self.layers.add_module('layer3', self._make_layer(block, 256, layers[2], stride))
        in_channels = 512
        stride = 2
        self.layers.add_module('layer4', self._make_layer(block, 512, layers[3], stride))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
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
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc(out)

        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        num_classes = int(out_shape[0])
        dropout = prm.get('dropout', 0.0)
        super(Net, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            in_shape=in_shape,
            out_shape=out_shape,
            num_classes=num_classes,
            dropout=dropout
        )
        self.device = device

    def forward(self, x):
        return super(Net, self).forward(x)

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