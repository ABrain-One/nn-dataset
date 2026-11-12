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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        # Build the feature extractor with standard layers
        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 64)
        self.layer3 = Block(64, 128)
        # Squeeze and expand for channel reduction and increase
        self.squeeze = nn.Conv2d(128, 32, kernel_size=1)
        self.expand = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        # Downsample for spatial reduction
        self.downsample = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        # Repeat and reduce blocks for feature density
        self.layer4 = Block(128, 128)
        self.layer5 = Block(128, 128)
        self.layer6 = Block(128, 128)
        self.layer7 = Block(128, 128)
        self.layer8 = Block(128, 128)
        # Final layer with standard configuration
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        # Efficient forward pass to reduce computation and latency
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.squeeze(x)
        x = self.expand(x)
        x = self.downsample(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = self.flatten(x)
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