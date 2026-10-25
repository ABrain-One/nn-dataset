import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = self.build_layer(32, 32, 3, stride=1)
        self.layer2 = self.build_layer(32, 64, 3, stride=2)
        self.layer3 = self.build_layer(64, 128, 3, stride=2)
        self.layer4 = self.build_layer(128, 128, 3, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = None

    def build_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = self.criteria.to(self.device)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

def check_shape(shape):
    assert len(shape) == 4
    assert shape[1] in {1, 3}
    assert all(dim % 2 == 0 for dim in shape[2:])

def get_workspace_size(size_limit):
    class Workspace(nn.Module):
        def __init__(self, size_limit):
            super().__init__()
            self.size_limit = size_limit
            self.register_buffer('buffer', torch.zeros(1))

        def forward(self, x):
            desired_size = x.numel()
            if desired_size > self.size_limit:
                raise ValueError("Workspace size limit exceeded.")
            self.buffer.resize_(desired_size)
            return x

    return Workspace(size_limit)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_epoch(net, train_data, prm):
    net.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(net.device), labels.to(net.device)
        net.optimizer.zero_grad()
        outputs = net(inputs)
        loss = net.criteria(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3)
        net.optimizer.step()

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.device = device
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.classes = out_shape[0]
        self.num_params = self.get_n_params()
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = self.build_classifier()
        self.workspace = get_workspace_size(500000 - self.num_params)
        self.optimizer = None
        self.criteria = None

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.layer1 = nn.Sequential(*layers)
        return self.layer1

    def build_classifier(self):
        return nn.Linear(128 * 6 * 6, self.num_classes)

    def get_n_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def forward(self, x):
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.workspace(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.criteria = nn.CrossEntropyLoss()
        self.to(self.device)

    def learn(self, train_data):
        train_epoch(self, train_data, prm={'lr': 0.01, 'momentum': 0.9})