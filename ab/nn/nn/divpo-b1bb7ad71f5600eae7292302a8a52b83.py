import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple[int], out_shape: tuple[int], prm: dict[str, float], device: torch.device) -> None:
        super().__init__()
        assert len(in_shape) == 4 and in_shape[1] == 3, f"Expected input shape to be (batch, 3, height, width)"
        self.device = device
        self.num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            self._make_layer(32, 32, 2),
            self._make_layer(32, 64, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.3)),
            nn.Linear(256, self.num_classes),
        )

    def _make_layer(self, in_planes: int, out_planes: int, blocks: int) -> nn.Sequential:
        layers = []
        stride = 1
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_planes))
        for _ in range(blocks):
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(out_planes))
            stride = 2
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm: dict[str, float]) -> None:
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