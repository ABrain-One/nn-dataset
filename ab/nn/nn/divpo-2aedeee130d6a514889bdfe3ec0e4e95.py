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
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = out_shape[0]  # number of target classes
        # First layer takes in_channels from in_shape:
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.ReLU(inplace=True)

        # Block with 32 output channels
        self.block1 = Block(32, 32)

        # Downsample to 64 output channels
        self.downsample1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # Blocks with 64 output channels
        self.block2 = Block(64, 64)
        self.block3 = Block(64, 64)

        # Downsample to 128 output channels
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # Blocks with 128 output channels
        self.block4 = Block(128, 128)
        self.block5 = Block(128, 128)

        # Adaptive average pool and classifier
        self.gap = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)

        # Blocks
        x = self.block1(x)

        # Downsample
        x = self.downsample1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Downsample
        x = self.downsample2(x)
        x = self.block4(x)
        x = self.block5(x)

        # Global average pooling
        x = self.gap(x)
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

# Ensure Net does not exceed parameter count:
if __name__ == '__main__':
    net = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')
    print(f'Total params: {sum(p.numel() for p in net.parameters())}')