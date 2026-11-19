import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.drop_prob = drop_prob

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        if self.drop_prob > 0:
            out = F.dropout(out, p=self.drop_prob, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3, drop_prob=0.0):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet blocks
        # CIFAR images are 32x32. We keep stride 1 in the stem to maintain 32x32 spatial dimensions.
        # Block 1: 32x32 -> 32x32 (stride 1)
        # Block 2: 32x32 -> 16x16 (stride 2)
        # Block 3: 16x16 -> 8x8 (stride 2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

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
        
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        # Use momentum as a proxy for regularization strength in this context, 
        # or simply map it to a small dropout rate if we want to use it for regularization.
        # However, standard ResNet doesn't use dropout heavily. 
        # To strictly satisfy the "use momentum" requirement in a meaningful way beyond the optimizer,
        # we can interpret a higher momentum as implying a need for slightly more regularization 
        # in the early stages of training, or simply use it to scale a small dropout.
        # Let's use a fixed small dropout of 0.1 for stability, but we must read momentum.
        # Actually, the prompt says "use only names the file already reads from prm".
        # The baseline reads lr and momentum.
        # I will use momentum in the optimizer (as required) and potentially use a fixed dropout.
        # Wait, the rule says: "each of those names ... must still be read from prm and actually used in the new model".
        # Using it in the optimizer counts as "used in the new model" (the Net class).
        # But to be safe and "actually used" in the architecture logic if possible, 
        # I'll stick to standard practice: momentum is for SGD.
        
        # We will build a ResNet-18 variant suitable for CIFAR-10.
        # Layers: [2, 2, 2]
        self.model = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2],
            num_classes=self.class_number,
            in_channels=self.channel_number,
            drop_prob=0.0  # We can add dropout if we wanted to use a hyperparam, but we don't have one for it.
        )

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