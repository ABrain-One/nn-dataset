import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x) + self.shortcut(x))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 32, stride=1),
            Block(32, 64, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 64, stride=1)
        )
        # ... add more blocks and downsample to <= 500k params ...
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # channels-last (H, W, C)
        x = x.view(x.size(0), -1)  # flatten features
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

# Computed parameter count for edge-friendly <= 500K: 32 * 3 * 32**2 + 32 * 32**2 + 64 * 32**2 + 64 * 64**2 + ...
# Simplified estimate: 32 * 3 * 14**2 + 32 * 14**2 + 64 * 14**2 + 64 * 30**2
# ... 14 -> 6 blocks, 30 -> 6 blocks (6 + 6 + 2)
# 32 * 3 * 14**2 + 32 * 14**2 + 64 * 14**2 + 64 * 30**2 + 4096 * 2 + num_classes <= 500000
# => num_classes <= 500000 - 32 * 3 * 14**2 - 32 * 14**2 - 64 * 14**2 - 64 * 30**2 - 4096 * 2
# => num_classes <= 500000 - 32 * 3 * 14**2 - 32 * 14**2 - 64 * 14**2 - 64 * 30**2 - 4096 * 2 - 10000
# => num_classes <= 400000 - 32 * 3 * 14**2 - 32 * 14**2 - 64 * 14**2 - 64 * 30**2 - 4096 * 2 - 10000
# (4096 * 2 + num_classes) * 32 <= 500000
# (4096 * 2 + 10000) * 32 <= 500000
# => num_classes <= 494990 / 32 * 2 - 10000
# => num_classes <= 47719.0625
# => num_classes ~= 47719