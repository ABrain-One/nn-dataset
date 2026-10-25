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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = nn.Sequential(
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            Block(32, 64, stride=1, groups=32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            Block(64, 128, stride=1, groups=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
model_dummy = Net(
    in_shape=[in_channels_dummy, in_height_dummy, in_width_dummy],
    out_shape=[out_channels_dummy,],
    prm={'lr': 0.01, 'momentum': 0.9},
    device='cpu'
)
model_dummy = model_dummy.to('cpu')
model_dummy.train()

# Actual model with derived channels and out_channels.
in_channels = in_channels_dummy
out_channels = out_channels_dummy
model = Net(
    in_shape=[in_channels, in_height_dummy, in_width_dummy],
    out_shape=[out_channels,],
    prm={'lr': 0.01, 'momentum': 0.9},
    device='cpu'
)
model.train_setup({'lr': 0.01, 'momentum': 0.9})

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
assert total_params <= 500000, "Total parameters must be within the limit"