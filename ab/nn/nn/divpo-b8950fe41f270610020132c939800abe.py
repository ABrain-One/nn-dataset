import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, groups=32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, groups=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            Block(128, 128, groups=128),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Example model structure with a single block for brevity (must be no more than 500,000 parameters):
# Net (in_shape: (3, 32, 32), out_shape: (10,), prm: {'lr': 0.01, 'momentum': 0.9}, device: device)
# - Module: 0 parameters, 0.00 M parameters
#   - Conv2d-1 (3, 32, 32, 32)
#   - BatchNorm2d-2 (32)
#   - ReLU-3
#   - Module: 320 parameters, 0.00 M parameters
#     - Conv2d-4 (32, 32, 3, 1, 1)
#     - BatchNorm2d-5 (32)
#     - ReLU-6
# - MaxPool2d-7 (32, 16, 16)
# - Module: 320 parameters, 0.00 M parameters
#   - Module: 320 parameters, 0.00 M parameters
#     - Conv2d-8 (32, 64, 3, 1, 1)
#     - BatchNorm2d-9 (64)
#     - ReLU-10
#     - Module: 320 parameters, 0.00 M parameters
#       - Conv2d-11 (64, 64, 3, 1, 1)
#       - BatchNorm2d-12 (64)
#       - ReLU-13
#   - MaxPool2d-14 (64, 8, 8)
# - Module: 320 parameters, 0.00 M parameters
#   - Module: 320 parameters, 0.00 M parameters
#     - Conv2d-15 (64, 128, 3, 1, 1)
#     - BatchNorm2d-16 (128)
#     - ReLU-17
#     - Module: 320 parameters, 0.00 M parameters
#       - Conv2d-18 (128, 128, 3, 1, 1)
#       - BatchNorm2d-19 (128)
#       - ReLU-20
#   - MaxPool2d-21 (128, 4, 4)
# - Module: 320 parameters, 0.00 M parameters
#   - Module: 320 parameters, 0.00 M parameters
#     - Conv2d-22 (128, 128, 3, 1, 1)
#     - BatchNorm2d-23 (128)
#     - ReLU-24
#     - Module: 320 parameters, 0.00 M parameters
#       - Conv2d-25 (128, 128, 3, 1, 1)
#       - BatchNorm2d-26 (128)
#       - ReLU-27
#   - AdaptiveAvgPool2d-28 (128, 6, 6)
# - Dropout-29 (0.5)
# - Linear-30 (9216, 512)
# - ReLU-31
# - Linear-32 (512, 10)