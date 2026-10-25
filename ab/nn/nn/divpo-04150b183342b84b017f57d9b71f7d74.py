import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(
            Block(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
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

# Edge-friendly latency budget: tight
# Params <= 500000
# Use standard layers only; no pretrained weights.
# Derive channels and input shape from in_shape as (C, H, W) channels-first.
# Supported hyperparameters: lr, momentum

# Auto-generated design criteria:
# Aspects:
#   - Use standard conv blocks: Block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
#   - Prefer batch norm: nn.BatchNorm2d
#   - Keep pre-activation: in blocks (relu before bn)
#   - Avoid se(module): do not use SE blocks
#   - Keep residual: do not use F(x+y) patterns
#   - Prefer depth to width: max pool after each block
#   - Use 2D conv only: no Attention, GELU, etc.
#   - Adhere to parameter count: <= 500000
#   - Robust to input size: should work with (3, 32, 32) and (3, 32, 32)

# Design target: achieve higher accuracy with up to 500000 parameters.