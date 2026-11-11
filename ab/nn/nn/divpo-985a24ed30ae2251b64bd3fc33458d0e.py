import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = Block(in_channels, 32, kernel_size=3, padding=1, stride=1)
        # Build feature extractor with 2x blocks:
        self.layer1 = Block(32, 32, kernel_size=3, padding=1, stride=1)
        self.layer2 = Block(32, 64, kernel_size=3, padding=1, stride=2)
        # Downsample for 2x block:
        self.layer3 = Block(64, 64, kernel_size=3, padding=1, stride=2)
        # Avg pool to keep spatial size low:
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        # Flatten and classifier:
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        # First block
        x = self.stem(x)
        # 2x blocks with downsampling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Avg pool and flatten
        x = self.avg_pool(x)
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