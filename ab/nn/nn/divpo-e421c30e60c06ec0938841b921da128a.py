import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, 3, kernel_size=3, padding=1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv2d(64, 6, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(6, 64, kernel_size=(7, 1), padding=(3, 0)),
            nn.GroupNorm(64 // 8, 64),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.GroupNorm(64 // 8, 64),
            nn.Mish(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.GroupNorm(16 // 8, 16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(16 // 8, 16),
            nn.Mish(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32 // 8, 32),
            nn.Mish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32 // 8, 32),
            nn.Mish(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(64 // 8, 64),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(64 // 8, 64),
            nn.Mish(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.blocks = nn.ModuleList([Block(64, 64, 1)])
        self.classifier = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()