import torch
from torch import nn, Tensor
from functools import partial


class ResidualBlock(nn.Module):
    """Basic ResNet block for CIFAR (3x3 convs, no downsampling in first block)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm_layer: nn.Module = None,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes = out_shape[0]

        # ResNet-9 for CIFAR: 3 stages with 1 block each, widths [64, 128, 256]
        # Widths are fixed to standard ResNet-9 (no width_mult scaling as per standard choice)
        widths = [64, 128, 256]

        norm_layer = partial(nn.BatchNorm2d, eps=prm['norm_eps'], momentum=prm['norm_momentum'])
        dropout = prm['dropout']

        # Initial conv: 3x3, stride 1 (no downsampling to preserve spatial info for CIFAR)
        self.conv1 = nn.Conv2d(in_shape[1], widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(widths[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 1 block, 64 channels, stride 1
        self.layer1 = ResidualBlock(widths[0], widths[0], stride=1, norm_layer=norm_layer, dropout=dropout)

        # Stage 2: 1 block, 128 channels, stride 2
        self.layer2 = ResidualBlock(widths[0], widths[1], stride=2, norm_layer=norm_layer, dropout=dropout)

        # Stage 3: 1 block, 256 channels, stride 2
        self.layer3 = ResidualBlock(widths[1], widths[2], stride=2, norm_layer=norm_layer, dropout=dropout)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(widths[2], num_classes)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'norm_eps', 'norm_momentum'}