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
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            self._make_layer(32, 32, 2),
            self._make_layer(32, 64, 2),
            self._make_layer(64, 64, 2),
        )
        self.classifier = nn.Linear(64, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, (7, 7))
        x = self._avg_pooling(x)
        x = self._linear(x)
        return x

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int) -> nn.Sequential:
        layers = []
        layers.append(self._bn_conv(in_planes, planes))
        for _ in range(1, num_blocks):
            layers.append(self._bn_conv(planes, planes))
        return nn.Sequential(*layers)

    def _bn_conv(self, in_planes: int, planes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def _avg_pooling(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.avg_pool2d(x, 7)

    def _linear(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.classifier(x)