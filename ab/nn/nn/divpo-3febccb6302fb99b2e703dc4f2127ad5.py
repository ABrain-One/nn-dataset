import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
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
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
        )
        self._build_layers(32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.train_setup(prm)

    def _build_layers(self, in_channels):
        self.layer1 = nn.Sequential(Block(in_channels, in_channels, stride=1))
        self.layer2 = nn.Sequential(
            Block(in_channels, in_channels, stride=2),
            Block(in_channels, in_channels),
        )
        self.layer3 = nn.Sequential(
            Block(in_channels, in_channels, stride=2, groups=in_channels),
            Block(in_channels, in_channels),
        )
        self.layer4 = nn.Sequential(
            Block(in_channels, in_channels, stride=2, groups=in_channels),
            Block(in_channels, in_channels),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer3(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer4(x)
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

# Dummy input to derive the correct in_channels and out_channels.
in_channels_dummy = 3
in_height_dummy = 32
in_width_dummy = 32
out_channels_dummy = 10
dummy_input = torch.randn(1, in_channels_dummy, in_height_dummy, in_width_dummy)
out_shape_dummy = (out_channels_dummy,)
in_shape_dummy = (in_channels_dummy, in_height_dummy, in_width_dummy)

# Actual model with the derived in_channels and out_channels.
model_candidate = Net(
    in_shape_dummy,
    out_shape_dummy,
    prm={'lr': 0.01, 'momentum': 0.9},
    device=torch.device('cpu'),
)
params = sum(p.numel() for p in model_candidate.parameters())
assert params <= 500000, "Total parameters must be within the limit"