import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: str) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self) -> nn.Sequential:
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers += self._make_layer(32, 32, 1, stride=1)
        layers += self._make_layer(32, 64, 2, stride=2)
        layers += self._make_layer(64, 128, 2, stride=2)
        layers += self._make_layer(128, 256, 2, stride=2)
        layers += self._make_layer(256, 512, 2, stride=2)
        self._last_channels = 512
        return nn.Sequential(*layers)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        for block_idx in range(blocks):
            stride = stride if block_idx == 0 else 1
            layers += [self._make_block(in_channels, out_channels, stride)]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        expansion = 1
        mid_channels = out_channels * expansion
        layers = []
        if stride != 1 or in_channels != out_channels:
            layers += [
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
            ]
        layers += [
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()