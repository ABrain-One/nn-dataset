import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = self._make_layer(32, 32, 2, 1)
        self.block2 = self._make_layer(32, 64, 2, 2)
        self.block3 = self._make_layer(64, 128, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = None

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = self.criteria.to(self.device)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_trace(model, x, prm=None):
    model = model.eval()
    model = torch.jit.trace(model, x)
    return model

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}