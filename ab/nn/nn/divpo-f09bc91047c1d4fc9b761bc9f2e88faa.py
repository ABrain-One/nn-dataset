import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        # Extract shape information from in_shape
        if len(in_shape) == 4:
            _, self.in_channels, self.height, self.width = in_shape
        else:
            self.in_channels = int(in_shape[0])
            self.height = int(in_shape[1])
            self.width = int(in_shape[2])

        self.num_classes = int(out_shape[0])

        # Initialize layers
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_layer(32, 32, 2),
            self._make_layer(32, 64, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.num_classes),
        )

    def _make_layer(self, in_planes: int, out_planes: int, num_blocks: int) -> nn.Sequential:
        layers = []
        layers.append(self._make_basic_block(in_planes, out_planes))
        for _ in range(1, num_blocks):
            layers.append(self._make_basic_block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _make_basic_block(self, in_planes: int, planes: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()