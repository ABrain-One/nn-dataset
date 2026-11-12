import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Block(32, 32),
            Block(32, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 64),
            Block(64, 128),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

# Edge-friendly, standard-layer only design inspired by efficient architectures.
# Structure follows a pattern of blocks with 1x1-conv SE blocks (no SELU) to reduce params.
# Emerge 32-64-128 feature channels after 1-2 blocks each (vs 32-32-64-128 in prior designs).
# Keep params under 500K and latency low by:
# 1. Using 1x1-conv SE blocks for channel reduction (avg 32->16 channels per block).
# 2. Avoiding SELU to keep activation count low.
# 3. Efficient block patterns: 32-32-128 (vs 32-32-32-64-128 in prior).
# 4. Use 3x3-conv instead of 5x5 (to avoid extra 1x1s).
# 5. Keep layers shallow (max 2 blocks per layer, no skip/skip-in).
# 6. Limit filters to 16, 32, 64, 128 (no 48, 80, 160).
# 7. Keep pre-activation for all blocks (no ResNet-style post-activation).
# 8. Use 1x1-conv SE blocks for channel reduction after each block.
# 9. Use 2-3 blocks per layer for depth 18-34 (vs 3-5).
# 10. Add dropout (0.5) after first linear for regularization.

# Example SE block (1x1-conv for channel reduction, no SELU).
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduce_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduce_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduce_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Target average params within 500000. Tune for minimum final layer size.

# Design guidelines:
# - Target params below 500000.
# - Achieve edge-friendly latency on modern hardware.
# - Use standard PyTorch layers only.
# - Do not use pretrained weights.
# - Use in_shape[1] for channels (e.g., 3 for RGB).
# - Derive from this block or create a new one if needed.