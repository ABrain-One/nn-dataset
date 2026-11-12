import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_se = se_ratio > 0
        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # SE
            nn.Sequential() if not self.use_se else nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            ),
            # Project
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.channels = 32
        self.stage_repeats = [3, 3, 3]
        self.bottleneck_repeats = [3, 3, 3]
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.channels, self.num_classes)

        self.dropout_p = prm.get('dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_p)
        self.momentum = prm.get('momentum', 0.9)
        self.learning_rate = prm.get('lr', 0.025)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU6(inplace=True),
        ]

        # MobileNetV2 style inverted residual blocks
        in_channels = self.channels
        for i, (t, c, n, s) in enumerate([(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2)]):
            layers += [InvertedResidual(in_channels, c, s, t)]
            in_channels = c
            for _ in range(n):
                layers += [InvertedResidual(in_channels, c, 1, t)]
                in_channels = c

        # Final pointwise convolution
        layers += [
            nn.Conv2d(in_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        ]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.025),
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']

def supported_hyperparameters():
    return ['lr', 'momentum']