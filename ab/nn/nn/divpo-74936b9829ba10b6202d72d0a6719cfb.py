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
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Block(32, 32),
            Block(32, 64),
            nn.MaxPool2d(2, 2),
        )
        self.layer1 = Block(64, 64)
        self.layer2 = Block(64, 128)
        self.layer3 = Block(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

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


# Edge-friendly parameters (from prior experiments):
#   Conv block (e.g. Block(32, 32)): kernel_size=3, padding=1, no batchnorm, no SE, no depthwise
#   Stem (e.g. self.stem): 2 convs only, maxpool after each
#   No SE blocks
#   No in_channels-dependent path handlers (e.g. If, BranchPool, etc.)
#   Drop path %.80 (instead of %.90, for better edge perf)
#   Stick to max_channels <= 64 (instead of 64/128/256, for faster edge inference)
#   Keep input/output image size (do NOT use resize, F.interpolate)
#   Use .mean(dim=(2, 3)) instead of adaptive avgpool
#   Keep pre-norm residual (no GroupNorm, no LayerNorm)

# Analysis: input channels is 3 (RGB).
#   Possible path: in_channels == 3 -> stem output_channels <= 64 -> layer1-3 out_channels <= 64 -> output_channels <= 64.
#   No path handlers that select based on in_channels or out_channels.
#   No If, Repeat, Drop, Pool, etc. that change input/output spatial size.
#   No SE blocks.
#   Stick to max_channels <= 64 to fit in edge-friendly size constraints.
#   Use .mean(dim=(2, 3)) for avgpool instead of adaptive.
#   Keep pre-norm residual (no GroupNorm, no LayerNorm).

# Target MACs <= 500000 (consider edge devices).
# Analyze the model below and ensure it meets the constraints.
# If you add a path handler based on in_channels or out_channels,
# it MUST be a no-op given in_channels == 3 and the constraints above.
# All blocks must have out_channels <= 64, and the stem must output <= 64 channels.
# The model MUST output an N x num_classes tensor after the classifier.