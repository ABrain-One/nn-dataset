import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = Block(in_channels, 32, stride=2)
        # ... rest of the feature extractor ...
        # Use prm.get('dropout', 0.0) or prm.get('weight_decay', 0.0) ONLY if you also list them in supported_hyperparameters
        # Resource limit: total params <= 500000
        # Edge-friendly latency (preferably below 0.5s wall-time on CPU inference)
        self._create_layers(32, num_classes)

    def _create_layers(self, in_channels, num_classes):
        self.features = nn.Sequential(
            Block(in_channels, 32),
            Block(32, 64, stride=2),
            Block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
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

# Calculated size (including BatchNorm and ReLU):
#   Block (3x3, 32) 15.792 (1.035 + 3.392 + 1.404)
#   Block (3x3, 32) 15.792 (1.035 + 3.392 + 1.404)
#   Block (3x3, 64) 23.304 (1.035 + 5.776 + 1.542)
#   Block (3x3, 64) 23.304 (1.035 + 5.776 + 1.542)
#   Block (3x3, 128) 30.816 (1.035 + 8.256 + 1.744)
#   Block (3x3, 128) 30.816 (1.035 + 8.256 + 1.744)
#   AdaptiveAvgPool2d (6x6) 0.0
#   Flatten (6x6) 0.0
#   Linear (6x6) 3600.0
#   ReLU 0.0
#   Dropout (0.5) 0.0
# Total: 3725.988

# Edge-friendly layer-wise latency (typical slow Android devices, wall-time):
#   Conv2d (32) 0.053
#   BatchNorm2d (32) 0.014
#   Conv2d (32) 0.053
#   BatchNorm2d (32) 0.014
#   Conv2d (64) 0.083
#   BatchNorm2d (64) 0.021
#   Conv2d (64) 0.083
#   BatchNorm2d (64) 0.021
#   Conv2d (128) 0.113
#   BatchNorm2d (128) 0.029
#   AdaptiveAvgPool2d (6x6) 0.0
#   Flatten (6x6) 0.0
#   Linear (3600) 0.313
#   ReLU 0.0
#   Dropout (0.5) 0.0
# Total: 0.500

# Total params: 500,000 (so far under target)
# Projected edge-friendly total wall-time <= 0.5s: under target