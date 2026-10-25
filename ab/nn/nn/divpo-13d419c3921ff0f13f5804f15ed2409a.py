import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self.build_features()
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # Make sure to keep _last_channels up to date in your block
        self._last_channels = 32

        layers += [
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Add your blocks here
            # Block1(in_channels=32, out_channels=32),
            # Block2(in_channels=32, out_channels=64),
            # ...

            nn.AdaptiveAvgPool2d((6, 6)),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class Block1(nn.Module):
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
        out = self.relu(out)
        return out

class Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

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