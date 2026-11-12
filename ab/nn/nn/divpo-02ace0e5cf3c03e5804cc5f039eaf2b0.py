import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels * expansion)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1),
            Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1)
        )
        self.layers = nn.Sequential(
            Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1),
            Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1),
            Block(32, 64, kernel_size=3, padding=1, stride=2, expansion=2),
            Block(64, 64, kernel_size=3, padding=1, stride=1, expansion=2),
            Block(64, 128, kernel_size=3, padding=1, stride=2, expansion=2),
            Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=2),
            Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=2),
            Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=2),
            Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=2),
            Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        # Move to device and create optimizer/criterion
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}