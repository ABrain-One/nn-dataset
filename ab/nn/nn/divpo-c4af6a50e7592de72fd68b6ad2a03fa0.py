import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

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
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.Sequential(
            Block(32, 32),
            Block(32, 32),
            nn.MaxPool2d(2, 2),
            Block(32, 64),
            Block(64, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            Block(128, 128),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

# Edge-friendly latency target (20ms on typical mobile devices)
# Do not exceed: total parameters <= 500000, all block/layer creations are efficient
# Optimize for minimum runtime post-inference setup (no warmup; results only on 1st pass).
# Resource analysis is done based on in_shape (C x H x W or batch x C x H x W).
# Derive all layer dimensions from in_shape; do NOT hardcode input size.
# Standard PyTorch layer rules apply (no APEX/SyncBN/etc.; use nn only).

# Auto-generated analysis: in_shape has 4 dimensions (batch-first).
# Derived constraints: input channels <= 3; layer stride <= 2.
# Resource target: params <= 500000.
# Suggested layer dimensions: channels <= 32, odd-sized (h x w) kernels <= 3 (prevents large feat maps).

# Design goal: achieve fastest latency (edge-friendly) while staying under the parameter limit.