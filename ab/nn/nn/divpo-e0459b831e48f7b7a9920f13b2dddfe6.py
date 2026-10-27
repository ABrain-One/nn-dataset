import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 32)
        self.layer3 = Block(32, 64)
        self.layer4 = Block(64, 64)
        self.layer5 = Block(64, 96)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(96 * 6 * 6, out_shape[0])

        self._make_layers(in_shape)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def _make_layers(self, in_shape: tuple) -> None:
        in_channels = in_shape[1]
        layers = []
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(Block(32, 32))
        layers.append(Block(32, 32))
        in_channels = 32
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(Block(32, 32))
        layers.append(Block(32, 32))
        in_channels = 32
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(Block(64, 64))
        layers.append(Block(64, 64))
        in_channels = 64
        layers.append(nn.Conv2d(in_channels, 96, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(Block(96, 96))
        layers.append(Block(96, 96))
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

class Block(nn.Module):
    """ Residual block w/ pre-norm """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out