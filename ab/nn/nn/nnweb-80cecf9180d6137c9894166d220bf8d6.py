import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResNet9Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet9Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_scale=0.75):
        super(ResNet9, self).__init__()
        
        # Base channels for ResNet-9
        base_channels = 64
        # Apply width scale
        c1 = int(base_channels * width_scale)
        c2 = int(base_channels * 2 * width_scale)
        c3 = int(base_channels * 4 * width_scale)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet-9 typically has 3 stages with 1 block each (total 3 residual blocks)
        # Stage 1: 32x32 -> 32x32 (stride 1)
        self.layer1 = ResNet9Block(c1, c1, stride=1)
        
        # Stage 2: 32x32 -> 16x16 (stride 2)
        self.layer2 = ResNet9Block(c1, c2, stride=2)
        
        # Stage 3: 16x16 -> 8x8 (stride 2)
        self.layer3 = ResNet9Block(c2, c3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # Initialize ResNet-9 with width scale 0.75 as mentioned in the description
        self.model = ResNet9(self.channel_number, self.class_number, width_scale=0.75)

        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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