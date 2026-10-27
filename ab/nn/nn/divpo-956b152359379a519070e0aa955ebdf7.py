import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        self._last_channels = 64
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(model, prm):
    model.to('cpu')
    model.criteria = (nn.CrossEntropyLoss(),)
    model.optimizer = torch.optim.SGD(
        model.parameters(),
        lr=prm['lr'],
        momentum=prm['momentum']
    )

def learn(model, train_data):
    model.train()
    for inputs, labels in train_data:
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criteria[0](outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        model.optimizer.step()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def create_model(in_shape, out_shape, prm):
    device = 'cpu'
    model = Net(in_shape, out_shape, prm, device)
    model.to(device)
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)