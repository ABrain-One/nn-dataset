import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(32, 32, 3, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2)
        self.layer3 = self._make_layer(64, 128, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

# Example of how to use the Net class:
def test_net():
    in_shape = (3, 32, 32)
    out_shape = (10,)
    prm = {'lr': 0.01, 'momentum': 0.9}
    device = torch.device('cpu')
    model = Net(in_shape, out_shape, prm, device)
    data = torch.randn(1, *in_shape)
    output = model(data)
    assert output.shape == torch.Size(out_shape)
    assert isinstance(model.criteria, nn.CrossEntropyLoss)
    assert isinstance(model.optimizer, torch.optim.SGD)
    assert model.optimizer.param_groups[0]['lr'] == prm['lr']
    assert model.optimizer.param_groups[0]['momentum'] == prm['momentum']