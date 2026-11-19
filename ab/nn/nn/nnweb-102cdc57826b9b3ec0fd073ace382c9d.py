import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device

        channel_number = in_shape[1]
        class_number = out_shape[0]
        dropout_p = prm['dropout']

        # CIFAR ResNet-18 Standard Layout
        # Stem: 3x3 conv, stride 1, no max-pool
        self.in_planes = 64
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Four stages of two BasicBlocks
        # Widths: 64, 128, 256, 512
        # Stride 2 in the first block of stages 2, 3, 4
        self.layer1 = self._make_layer(64, 2, stride=1, dropout=dropout_p)
        self.layer2 = self._make_layer(128, 2, stride=2, dropout=dropout_p)
        self.layer3 = self._make_layer(256, 2, stride=2, dropout=dropout_p)
        self.layer4 = self._make_layer(512, 2, stride=2, dropout=dropout_p)

        self.classifier = nn.Linear(512, class_number)

    def _make_layer(self, plane, blocks, stride, dropout):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride, dropout))
            self.in_planes = plane * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global Average Pooling
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        
        logits = self.classifier(out)
        return logits

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)

        lr = prm['lr']
        momentum = prm['momentum']

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()