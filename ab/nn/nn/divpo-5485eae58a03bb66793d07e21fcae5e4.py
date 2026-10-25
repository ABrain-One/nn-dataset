import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.eps = 1e-3

        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        for _ in range(3):
            self.features.add_module(
                "unit{}".format(_ + 3),
                InvertedResidual(64, 64, 1, 6)
            )
        self.features.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(64, out_shape[0])

        for _ in range(3):
            self.features.add_module(
                "unit{}".format(_ + 6),
                InvertedResidual(64, 64, 1, 6)
            )

        for _ in range(3):
            self.features.add_module(
                "unit{}".format(_ + 9),
                InvertedResidual(64, 64, 1, 6)
            )

        for _ in range(3):
            self.features.add_module(
                "unit{}".format(_ + 12),
                InvertedResidual(64, 64, 1, 6)
            )

        for _ in range(3):
            self.features.add_module(
                "unit{}".format(_ + 15),
                InvertedResidual(64, 64, 1, 6)
            )

        self.features.add_module(
            "unit{}".format(18),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.features.add_module("relu", nn.ReLU(inplace=True))
        self._set_classifier(out_shape)

    def _set_classifier(self, out_shape):
        if isinstance(out_shape, tuple):
            num_classes = out_shape[0]
        else:
            num_classes = out_shape
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()