import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2x3x3 convs) for CIFAR-scale images."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        out = F.relu(out, inplace=True)
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

        # ResNet-9: 1 initial conv + 3 stages of 1 block each (total 9 layers)
        # Standard CIFAR ResNet widths: [64, 128, 256]
        # "Configurable width" mentioned in description usually implies a multiplier, 
        # but without a specific 'width' hyperparameter in supported_hyperparameters, 
        # we use the standard base widths for ResNet-9 (which are typically 64, 128, 256).
        # If a width multiplier was intended, it's not in the fixed hyperparameters, 
        # so we stick to the standard architecture definition for ResNet-9.
        
        # Initial convolution
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 1: 1 block, 64 channels, stride 1
        self.layer1 = self._make_layer(64, 64, 1, stride=1, dropout=dropout)
        
        # Stage 2: 1 block, 128 channels, stride 2
        self.layer2 = self._make_layer(64, 128, 1, stride=2, dropout=dropout)
        
        # Stage 3: 1 block, 256 channels, stride 2
        self.layer3 = self._make_layer(128, 256, 1, stride=2, dropout=dropout)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, class_number)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout))
        in_channels = out_channels
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(in_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

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