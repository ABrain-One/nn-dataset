import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # Handle the channel dimension correctly
        self.num_classes = out_shape[0]

        self.features = nn.Sequential(
            Block(self.in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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

# Auto-generated compile target for edge-friendly latency
# Edge-friendly latency target: 1000.0ms (300.0ms + warmup 100.0ms)
# Optimize for minimum latency with no more than 500000 total parameters.
# Pruning or layer skipping allowed; prefer keeping initial channel count.

# Auto-generated design constraints
# Constraints:
# - Use standard PyTorch layers only, no custom classes.
# - Maximum total parameter count: 500000
# - Edge-friendly latency target: <= 1000.0ms
# - Optional warmup budget: 100.0ms for fast initialization
# - No channel pruning or layer skipping outside this block.
# - Include a skip connection if in_shape[1] != initial block's output channels.
# - Derive block's output channels from in_shape[1] (e.g., Block(in_channels, out_channels)).
# - Do NOT hardcode channel counts (out_channels) outside this block.

# Proposed architecture:
# Layer type guidelines: prefer Conv2d (no ConvTranspose2d, no nn.Linear layers outside).
# Keep parameter count below 500000 and aim for fast edge-friendly latency.
# Use in_shape[1] to derive the first block's input channels.
# Include a skip connection if in_channels != Block's output channels.

# Example of using in_shape[1] for the first block:
# self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)

# Target latency (combined warmup+train) <= 1000.0ms. Tune for fastest initialization and convergence.

# This is a dummy target for analysis; the actual task requires designing a unique network.