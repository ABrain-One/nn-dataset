import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            Block(32, 32)
        )

        self.layer1 = nn.Sequential(
            Block(32, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            Block(32, 64, kernel_size=3, stride=2, padding=1, groups=32),
            Block(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=2, padding=1, groups=64),
            Block(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
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

# Edge-friendly latency budget (suggested parameters)
# - Batch norm, depth-wise convs, or pointwise convs may make latency worse.
# - Avoid large kernels (3x3, 5x5) to keep params low.
# - Keep stride=1 in blocks to maintain resolution.
# - Depth-wise convs (groups>1) may add latency, keep groups=1 unless you study this.
# - Use in_channels directly rather than in_shape[0].
# - Keep params under 500k, but favor performance (e.g., more blocks, larger blocks).
# - Use an efficient model design that balances params and latency.
# - Consider testing block/layer combinations: 1-3x3-3, 1-3x3-2-3x3-3, 1-3x3-2-3x3-2-3x3-3.