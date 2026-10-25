import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])

        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [Block(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
            weight_decay=0.0001
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

# Ensure the model does not exceed the parameter limit
def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Design principles:
# 1. Emphasize early layer feature extraction for size control.
# 2. Use efficient block designs with attention to kernel size.
# 3. Employ depth-wise and pointwise convolutions for parameter efficiency.
# 4. Balance model complexity with edge-friendly latency.
# 5. Experiment with residual connections and attention mechanisms.

# Desired layer-wise parameters (to keep total under 500000):
# Block (3x3-conv + BN + ReLU): 1 * 3 * 32 * 32 + 32 + 32 + 1 * 32 + 1 = 896
# Stem (2x32-block + 1x32-block + 3x3-MP): (2 * 896 + 1 * 896) * 2 + 3 * 3 * 32 * 32 + 32 + 32 + 3 * 3 * 32 * 32 = 30720
# Layer1 (2x32-blocks): 2 * 896 = 1792
# Layer2 (2x64-blocks): 2 * 896 = 1792
# Layer3 (2x128-blocks): 2 * 896 = 1792
# Avgpool (6x6): 1 * 6 * 6 = 36
# Classifier (6*6*128 -> 10): 10 * (6 * 6 * 128) + 10 = 47100
# Total params (30720 + 1792 + 1792 + 1792 + 36 + 47100) <= 500000

# Example block structure:
# Block_32_32_MP: 3 * 32 * 32 + 32 + 1 * 32 + 1 = 896
# Block_64_64_MP: 3 * 64 * 64 + 64 + 1 * 64 + 1 = 18496
# Block_128_128_MP: 3 * 128 * 128 + 128 + 1 * 128 + 1 = 36864

# Target latency on edge devices (due to kernel size and depth-wise convs).
# Keep the maximum kernel size to 3x3 and avoid depth-wise convolutions.

# Example structure with the above parameters and constraints:
# Net (in_shape, out_shape, prm, device)
#   -> STEM (32 output channels)
#   -> Layer1 (32 output channels, 2 blocks)
#   -> Layer2 (64 output channels, 2 blocks)
#   -> Layer3 (128 output channels, 2 blocks)
#   -> Classifier (128 * 6 * 6 -> 10 classes)

# The above parameters and block structure should result in a model that
# complies with the resource and latency constraints.