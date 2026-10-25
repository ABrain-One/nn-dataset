import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = Block(64, 64, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = Block(128, 128, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
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
        x = self.avgpool(x)
        x = x.flatten(1)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Dummy input to derive the correct in_channels and out_channels.
dummy_input = torch.randn(1, 3, 32, 32)
in_channels = dummy_input.shape[1]
# Selected architecture: ResNet-like (repeat Block patterns)
# Device: cpu
# Previous layers (if any): 
# Architecture Guide: Repeat the following block pattern up to parameter counts. Start with 32-64-64, then:
# - After the first layer, each block alternates between:
#   - 1x3x64 (main), 1x1x64 (skip), and 2x3x64
#   - 1x3x64 (main) followed by 1x3x64 (middle), 1x1x64 (skip)
# - All use BatchNorm, ReLU, and avoid SE blocks.
# - Keep total params below 500000. No SE blocks, so channel counts are:
#   - After stem: 32->32->64
#   - After 1st block: 64->64->64
#   - After 2nd block: 64->128->128
#   - After 3rd block: 128->128->128
#   - After 4th block: 128->128->128
#   - AVG pool reduces 128x6x6 to 128x6x6.
#   - Classifier before final AVG pool: 128x6x6 -> 128x6x6 -> 64x3x3 -> 10 (no SE => out channels remain 10)
# Powered by AI. Evaluation should be on model size during forward pass.