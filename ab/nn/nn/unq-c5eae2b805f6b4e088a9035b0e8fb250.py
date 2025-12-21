import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # Define a simple block with depth-wise separable convolutions
        class DepthwiseSeparableBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super(DepthwiseSeparableBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        # Add DepthwiseSeparableBlock to the feature extractor
        layers += [
            DepthwiseSeparableBlock(32, 32, stride=1),
            DepthwiseSeparableBlock(32, 32, stride=1),
        ]

        # Define a simple block with depth-wise separable convolutions
        class DepthwiseSeparableBlock2(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super(DepthwiseSeparableBlock2, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        # Add DepthwiseSeparableBlock2 to the feature extractor
        layers += [
            DepthwiseSeparableBlock2(32, 32, stride=1),
            DepthwiseSeparableBlock2(32, 32, stride=1),
        ]

        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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