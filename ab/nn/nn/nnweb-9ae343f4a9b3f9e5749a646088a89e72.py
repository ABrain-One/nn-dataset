import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResBlock(nn.Module):
    """
    Basic residual block with batch normalization and ReLU.
    Includes a 1x1 projection shortcut when input/output channels or spatial size change.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0:
            identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        # CIFAR-10 is 32x32. We use a ResNet-18 style architecture adapted for 32x32 inputs.
        # Standard ResNet uses a 7x7 conv + maxpool. For CIFAR, we use a 3x3 conv (stride 1) 
        # and then 3x3 convs with stride 2 in the first block of each stage.
        
        # Stem
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 32x32 -> 32x32, 64 channels
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, dropout=dropout)
        
        # Stage 2: 32x32 -> 16x16, 128 channels
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, dropout=dropout)
        
        # Stage 3: 16x16 -> 8x8, 256 channels
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, dropout=dropout)
        
        # Stage 4: 8x8 -> 4x4, 512 channels
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, dropout=dropout)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, class_number)
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
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