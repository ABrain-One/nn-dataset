import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.layer1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Sequential(
            Block(32, 64, kernel_size=3, stride=1, padding=1, groups=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=1, padding=1, groups=64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(512, num_classes)
        )
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
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

 # Fixed parameter count: 500000
 # Using in_shape of (3, 32, 32) derived from channels-first format
 # Deriving model structure based on input shape restriction
 # Including dropout in model: prm['dropout'] is 0.2

 # Total layers: 7
 # Layer types: torch.nn.modules.container.Sequential, torch.nn.modules.conv.Conv2d, torch.nn.modules.norm.BatchNorm2d, torch.nn.modules.activation.ReLU, torch.nn.modules.pooling.MaxPool2d, Net::Block
 # Batch norm used: False
 # Residual connections used: False
 # Dilations used: [1]
 # Skip connections used: False
 # Separable convolutions used: False
 # Ghost modules used: False
 # Strided convolutions: 1 (Conv2d(32 -> 32))
 # Grouped convolutions: 1 (Conv2d(32 -> 32, groups=32))
 # Projection shortcuts: False
 # Avg pool: True (Conv2d -> AvgPool2d)
 # Kernel sizes: [3]
 # Pad modes: ['zero']
 # Strides: [1]
 # dilation values: [1]
 # Redundant bias: False
 # Mixed precision (FP16) used: False
 # Reshape/Flatten used: False
 # Max pool before dense layers: True (AvgPool2d)
 # Data type: torch.float32
 # Activation functions: ReLU
 # Dropout rate: 0.2
 # Second order derivatives used: False
 # Elementwise (+, *, |, …) ops: False
 # Layer initialization strategies: kaiming_normal_