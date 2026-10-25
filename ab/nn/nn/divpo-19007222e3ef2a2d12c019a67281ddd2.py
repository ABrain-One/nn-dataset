import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = Block(self.in_channels, 32, stride=2)
        self.layer1 = Block(32, 64, stride=2)
        self.layer2 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer2(x)
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
            momentum=prm.get('momentum', 0.9)
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

params = 500000

# Design principles:
# 1. Keep the model shallow to stay below the parameter limit.
# 2. Use stride to create spatial downsampling to keep spatial dims low.
# 3. Use batchnorm for better training stability and early convergence.
# 4. Use dropout or max pooling to further reduce spatial size for edge-friendliness.

# Example layer configuration: Block(32 -> 64, downsamples), Block(64 -> 128, downsamples), AvgPool -> GlobalAvgPool.

# Auto-generated design based on the criteria above (params <= 500000):

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # Derive input channels correctly
        self.num_classes = out_shape[0]
        self._layers = []

        # Block 1
        self._layers.append(Block(self.in_channels, 32, kernel_size=3, padding=1))
        self._layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # Block 2
        self._layers.append(Block(32, 64, kernel_size=3, padding=1, stride=2))
        self._layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # Adaptive Avg Pool and Classifier
        self.features = nn.Sequential(*self._layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
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
            momentum=prm.get('momentum', 0.9)
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

    def _get_name_postfix(self):
        # Example postfix for Block: "_32s2_3x3_32" for Block(32, 32, kernel_size=3, stride=2).
        postfix = f"{self._layers[0].in_channels}s{self._layers[0].out_channels}"
        for l in self._layers[1:]:
            postfix += f"_{l.kernel_size[0]}x{l.kernel_size[1]}_{l.out_channels}"
        return postfix

    def __str__(self):
        name = type(self).__name__
        postfix = self._get_name_postfix()
        return f"{name}_{postfix}"

supported_hyperparameters = {'lr', 'momentum'}

# Example layer structure:
# Block(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
# MaxPool2d(kernel_size=2, stride=2, padding=0)
# Block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
# MaxPool2d(kernel_size=2, stride=2, padding=0)
# AdaptiveAvgPool2d(output_size=(6, 6))
# Linear(in_features=192, out_features=num_classes, bias=True)

# Calculated params:
# (32s2_3x3_32 + 32s32_3x3_64 + 2maxpool + 6x6avgpool + 192xnum_classes_Linear) <= 500000