import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_shortcut=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, num_classes, in_channels, width_scale=0.75):
        super(ResNet9, self).__init__()
        base_width = int(64 * width_scale)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        
        # ResNet-9 typically consists of 3 stages with 1 block each, or similar compact structure.
        # Standard ResNet-9 for CIFAR often uses [1, 1, 1] blocks with strides [1, 2, 2].
        # Let's implement the standard compact layout:
        # Stage 1: 1 block, stride 1
        # Stage 2: 1 block, stride 2
        # Stage 3: 1 block, stride 2
        
        # Stage 1
        self.layer1 = ResidualBlock(base_width, base_width, stride=1, use_shortcut=False)
        
        # Stage 2
        self.layer2 = ResidualBlock(base_width, base_width * 2, stride=2, use_shortcut=True)
        
        # Stage 3
        self.layer3 = ResidualBlock(base_width * 2, base_width * 4, stride=2, use_shortcut=True)
        
        self.classifier = nn.Linear(base_width * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        self.model = ResNet9(num_classes=class_number, in_channels=channel_number, width_scale=0.75)
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

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