import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int, int], out_shape: tuple[int], prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_layer(32, 32, 1)
        self.block2 = self._make_layer(32, 64, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layer(self, in_planes: int, out_planes: int, num_blocks: int) -> nn.Sequential:
        layers = []
        layers.append(self._make_basic_block(in_planes, out_planes, num_blocks))
        return nn.Sequential(*layers)

    def _make_basic_block(self, in_planes: int, out_planes: int, num_blocks: int) -> nn.Sequential:
        strides = [1] + [2] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self._make_block(in_planes, out_planes, stride))
        return nn.Sequential(*layers)

    def _make_block(self, in_planes: int, out_planes: int, stride: int) -> nn.Sequential:
        padding = 1 if stride == 2 else 0
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        ]
        layers.append(nn.MaxPool2d(2, 2)) if stride == 2 else layers
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()