import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

        if len(in_shape) == 4:  # in_shape is (B, C, H, W) for CIFAR-10
            in_channels = in_shape[1]  # channels-first (C, H, W)
        else:  # in_shape is (C, H, W) channels-first
            in_channels = int(in_shape[0])

        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.stem.weight.data.normal_(0, 0.1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64)
        self.downsample = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample.weight.data.normal_(0, 0.1)
        self.blocks = self._make_blocks(64, 64, 1, 5)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def _make_blocks(self, in_channels, out_channels, layers_per_block, blocks_per_group):
        layers = []
        for _ in range(layers_per_block):
            for _ in range(blocks_per_group):
                layers.append(Block(in_channels, out_channels))
            if in_channels != out_channels:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
                layers[-1].weight.data.normal_(0, 0.1)
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.downsample(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
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

# Optional: helper blocks AFTER Net definition
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)