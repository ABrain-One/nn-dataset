import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

        # CIFAR ResNet-18: 3x3 stem, no max-pool
        self.stem = nn.Sequential(
            nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Four stages of two BasicBlocks: widths 64, 128, 256, 512
        # Stride 1 in stage 1, stride 2 in first block of stages 2-4
        self.layer1 = self._make_layer(64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, 2, stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, class_number)

    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = []
        in_channels = 64 if out_channels == 64 else 64 * (out_channels // 64) // 2
        # Correctly determine in_channels based on previous stage
        if out_channels == 64:
            in_channels = 64
        elif out_channels == 128:
            in_channels = 64
        elif out_channels == 256:
            in_channels = 128
        elif out_channels == 512:
            in_channels = 256

        layers.append(BasicBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'],)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

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