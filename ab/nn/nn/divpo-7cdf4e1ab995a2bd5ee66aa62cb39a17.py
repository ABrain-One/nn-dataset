import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.stem.to(device)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_bn.to(device)
        self.add_module('stem_act', nn.ReLU(inplace=True))

        # Repeat blocks to keep the total number under 500k parameters:
        blocks = []
        for _ in range(4):
            blocks.append(Block(32, 32))
        self.body = nn.Sequential(*blocks)
        self.body.to(device)

        # Downsample to 32x32:
        self.down1 = nn.MaxPool2d(2, 2, padding=0)
        self.down2 = nn.MaxPool2d(2, 2, padding=0)

        # Rest of the feature extractor (down to 6x6, then avgpool to 1x1):
        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 32)
        self.layer3 = Block(32, 32)
        self.layer4 = Block(32, 32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Single-cell classifier (no skip! keep size down):
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.classifier.to(device)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)

        # Downsample & body
        x = self.down1(x)
        x = self.body(x)
        x = self.down2(x)

        # Rest of the feature extractor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Avgpool & classify
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.flatten(1)  # flatten spatial dims
        return x

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

# Block is defined here, as is supported_hyperparameters.
# Keep this block unchanged:
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.functional.relu(x, inplace=True)