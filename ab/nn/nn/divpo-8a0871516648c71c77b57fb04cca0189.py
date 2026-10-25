import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_channels, hidden_dim, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise conv
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(in_channels, 16, stride=2),
            InvertedResidual(16, 16, stride=1, expand_ratio=3),
            InvertedResidual(16, 24, stride=2, expand_ratio=3),
            InvertedResidual(24, 24, stride=1, expand_ratio=3),
            InvertedResidual(24, 32, stride=2, expand_ratio=3),
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
            InvertedResidual(32, 64, stride=2, expand_ratio=4),
            InvertedResidual(64, 64, stride=1, expand_ratio=4),
            InvertedResidual(64, 96, stride=1, expand_ratio=4),
            InvertedResidual(96, 96, stride=1, expand_ratio=4),
            InvertedResidual(96, 160, stride=2, expand_ratio=4),
            InvertedResidual(160, 160, stride=1, expand_ratio=4),
            InvertedResidual(160, 320, stride=1, expand_ratio=4),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.mobilenetv3 = MobileNetV3(32, num_classes)
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.mobilenetv3(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()