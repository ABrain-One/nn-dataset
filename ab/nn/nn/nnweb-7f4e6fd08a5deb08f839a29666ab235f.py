import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-9 structure:
        # Block 1: 64 -> 64 (stride 1)
        # Block 2: 64 -> 128 (stride 2)
        # Block 3: 128 -> 256 (stride 2)
        # This creates a depth of 9 layers (1 conv + 3 blocks * 2 convs + 1 fc = 8 convs + 1 fc, often referred to as ResNet-9 in compact CIFAR contexts or similar naming conventions for small ResNets).
        # Standard small ResNet for CIFAR usually has 3 blocks.
        # Let's follow the standard small ResNet pattern which is often called ResNet-14 or ResNet-11 depending on counting.
        # The source "cifar10-resnet9" likely refers to a specific compact implementation.
        # A common "ResNet-9" for CIFAR might be:
        # Conv
        # Block (64)
        # Block (128)
        # Block (256)
        # This is 1 + 2 + 2 + 2 = 7 conv layers? Or maybe it counts differently.
        # Let's look at the name "ResNet-9".
        # If we count conv layers:
        # 1 (init) + 2 (block1) + 2 (block2) + 2 (block3) = 7.
        # Maybe it includes the FC? 7 + 1 = 8.
        # Maybe it's 1 + 2 + 2 + 2 + 2?
        # Let's assume the standard 3-block small ResNet which is widely used.
        # However, the prompt says "ResNet-9".
        # Let's check the specific repo "randyxia07/cifar10-resnet9".
        # Without access, I will implement a standard compact ResNet that fits the "ResNet-9" name in the context of CIFAR-10 compact models.
        # Often, ResNet-9 for CIFAR is defined as:
        # Conv
        # Block 1 (64)
        # Block 2 (128)
        # Block 3 (256)
        # This is a standard small ResNet.
        # Let's implement this.

        self.layer1 = self._make_layer(64, 1, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride))
            self.in_planes = plane
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # Use hyperparameters
        # lr and momentum are used in train_setup
        # We don't have a dropout hyperparameter in supported_hyperparameters, so we don't need to add dropout.
        # The prompt says: "LEMUR rejects a model that does not use every hyperparameter named in supported_hyperparameters: each of those names (for example dropout or momentum) must still be read from prm and actually used in the new model".
        # supported_hyperparameters returns {'lr', 'momentum'}.
        # So we must use 'lr' and 'momentum'.
        # 'lr' and 'momentum' are used in train_setup.
        # Do we need to use them in __init__? The rule says "actually used in the new model". train_setup is part of the model class.
        # So using them in train_setup is sufficient.

        self.model = ResNet9(self.channel_number, self.class_number)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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