import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2, dropout=0.2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2, dropout=0.3)
        self.layer3 = self._make_layer(64, 128, 3, stride=2, dropout=0.4)
        self.layer4 = self._make_layer(128, 256, 3, stride=2, dropout=0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = None

    def _make_layer(self, in_planes, out_planes, blocks, stride=1, dropout=0.):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_shape(shape):
    p = shape.split('x')
    values = list(map(int, p))
    return all(v > 0 for v in values)

class NetWrapper(nn.Module):
    def __init__(self, net: Net, in_shape: str, out_shape: list, prm: dict, device: torch.device):
        super().__init__()
        self.net = net
        self.in_channels = int(in_shape.split('x')[1])
        self.image_size = (int(in_shape.split('x')[2]), int(in_shape.split('x')[3]))
        self.num_classes = len(out_shape)
        self.device = device

    def forward(self, x):
        # NOTE: this forward function is ONLY for the case when the input is a batch of images.
        assert len(x.shape) == 4, 'Expecting a batch of images as input.'
        assert x.shape[1] == self.in_channels, 'Expecting the input channels to be {}'.format(self.in_channels)
        assert all(x.shape[2:] == self.image_size), 'Expecting the input image size to be {}x{}'.format(self.image_size[0], self.image_size[1])
        return self.net(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()