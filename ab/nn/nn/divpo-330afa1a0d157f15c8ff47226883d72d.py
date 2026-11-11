import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Helper block for residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != self.expansion * out_channels:
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

        if hasattr(self, 'shortcut'):
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out

# Residual Network Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_shape, out_shape, num_classes=10, dropout=0.0):
        super().__init__()
        in_channels = in_shape[1]
        self.device = in_shape[0].device if isinstance(in_shape[0], torch.Tensor) else None

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential()
        strides = [2, 1, 1, 1]
        in_channels = 64
        for i, num_block in enumerate(num_blocks):
            stride = strides[i]
            self.layers.append(self._make_layer(block, 64 * block.expansion, num_blocks=i, stride=stride))
        in_channels = 64 * block.expansion

        if in_channels != 512:
            self.layers.append(nn.Conv2d(in_channels, 512 * block.expansion, kernel_size=1, bias=False))
            self.layers.append(nn.BatchNorm2d(512 * block.expansion))

        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layers(out)

        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class Net(ResNet):
    def __init__(self, in_shape, out_shape, prm, device):
        num_blocks = [2, 2, 2, 2]  # Example for ResNet18
        num_classes = int(out_shape[0])
        dropout = prm.get('dropout', 0.0)
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        super().__init__(BasicBlock, num_blocks, in_shape, out_shape, num_classes, dropout)
        self.device = device

    def forward(self, x):
        return super().forward(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()