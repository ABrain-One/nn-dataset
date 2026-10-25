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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Block(32, 32),
            Block(32, 32),
        )
        self.layer1 = nn.Sequential(
            Block(32, 32),
            nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(32)),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            Block(32, 64),
            Block(64, 64),
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(64)),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            Block(64, 128),
            Block(128, 128),
            Block(128, 128),
            nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(128)),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        self._to_device(device)
        self._configure_optimizer(prm)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _to_device(self, device):
        self.to(device)

    def _configure_optimizer(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def train_setup(self, prm):
        self.train()
        self._to_device(self.device)
        self._configure_optimizer(prm)

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()