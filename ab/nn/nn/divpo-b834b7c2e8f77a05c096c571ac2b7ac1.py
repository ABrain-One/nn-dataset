import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(32, 32, 2, dropout=prm.get('dropout', 0.0))
        self.layer2 = self._make_layer(32, 64, 2, stride=2, dropout=prm.get('dropout', 0.0))
        self.layer3 = self._make_layer(64, 128, 2, stride=2, dropout=prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, blocks, stride=1, dropout=0.0):
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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

# Edge-friendly latency budget: tight
# Optimize for minimal runtime after training on CIFAR-10.
# The following guidelines are strictly adhered to:
# - Do not use advanced techniques like layer-wise mixed precision
# - No unused layers or channels
# - No redundant layer combinations (e.g., two sequential layers of the same type)
# - No pre- and post-processing outside the forward() path
# - Keep the total parameter count below 500000

# Design principles:
# - Focus on latency: optimize for minimal runtime after training
# - Use standard layers only
# - Derive block designs from the given Net class template
# - Respect resource limits: params <= 500000
# - Adhere to in_shape guidelines for input channel handling

# Analysis of the dataset:
# - Channels: 3 (RGB)
# - Image size: 32x32 (C x H x W)
# - Target size after models: 6x6 (H x W)

# Example of a fast, single-path block:
#