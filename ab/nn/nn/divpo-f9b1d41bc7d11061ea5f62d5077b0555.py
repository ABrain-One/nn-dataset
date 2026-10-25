import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._last_drops = None
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_layer(self, in_channels, out_channels, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def build_optimizer(self, prm, parameters):
        return torch.optim.SGD(
            parameters,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

def check_shape(shape):
    return len(shape) == 4 and shape[1] in {1, 3}

def patch_to_2d(m):
    if not isinstance(m, (nn.Conv1d, nn.Linear)): return m
    if m.weight.shape[1] == 1:
        m.weight = nn.Parameter(m.weight.squeeze(1))
        if m.bias is not None: m.bias = nn.Parameter(m.bias.squeeze(0))
    return m

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.in_shape: tuple = in_shape
        self.out_shape: tuple = out_shape
        self.device: torch.device = device
        self.channels: int = in_shape[1]
        self.image_size: int = in_shape[2]
        self.num_classes: int = out_shape[0]
        self.model: nn.Sequential = self.build_model()

    def build_model(self):
        layers: list = [
            nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.num_classes)
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_optimizer(self, prm, parameters):
        return torch.optim.SGD(
            parameters,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=5e-4
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def check_shape(self, shape):
        return len(shape) == 4 and shape[1] in {1, 3}

    def patch_to_2d(self, m):
        if not isinstance(m, (nn.Conv1d, nn.Linear)): return m
        if m.weight.shape[1] == 1:
            m.weight = nn.Parameter(m.weight.squeeze(1))
            if m.bias is not None: m.bias = nn.Parameter(m.bias.squeeze(0))
        return m

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_loss_fn(self, loss_name: str, prm: dict | None = None):
        if not prm: prm = {}