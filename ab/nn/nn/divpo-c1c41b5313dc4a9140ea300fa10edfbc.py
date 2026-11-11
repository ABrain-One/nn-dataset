import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = Block(in_channels, 32)
        # ... rest of the feature extractor ...
        self.blocks = nn.Sequential(
            Block(32, 32),
            Block(32, 32),
            Block(32, 64),
            Block(64, 64),
            Block(64, 128),
        )
        # ... add more blocks to reach max_params ...
        # Average pool to keep output shape manageable
        self.gap = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        # Use prm.get('dropout', 0.0) or prm.get('weight_decay', 0.0) ONLY if you also list them in supported_hyperparameters

    def forward(self, x):
        # Forward pass
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = x.flatten(1)  # flatten after average pool
        return self.classifier(x)

    def train_setup(self, prm):
        # Move to device and create optimizer/criterion
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

# Estimation of params (please adjust to your limit and count them in the Net constructor):
# (block * 32 * 32 + 32 * 32 + 32 + 64 * 64 + 64 * 64 + 64 + 128 * 6 * 6 + 128 * 6 * 6 + 32 * 128 * 6 * 6 + 128 * 6 * 6 + 64 * 6 * 6 + 64 * 6 * 6 + 32 * 6 * 6 + 6 * 6 * 6 * 64 + 64 + 128 + 128 + 32 + 64 + 32 + 64 + 32) * 2 / 1e6