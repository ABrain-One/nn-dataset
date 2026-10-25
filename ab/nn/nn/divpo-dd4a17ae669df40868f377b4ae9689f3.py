import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])
        self.optimizer = None
        self.criteria = None

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            self.optimizer.step()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum', 'weight_decay'}

def train_setup(self, prm):
    self.device = torch.device('cpu')
    self.model = Net(self.in_shape, self.out_shape, prm, self.device)
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=prm.get('lr', 0.01),
        momentum=prm.get('momentum', 0.9),
        weight_decay=prm.get('weight_decay', 0.0005)
    )
    self.criteria = (nn.CrossEntropyLoss().to(self.device),)

def learn(self, train_data):
    self.model.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criteria[0](outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
        self.optimizer.step()

def count_params(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

def validate(self, val_data):
    self.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def test(self, test_data):
    self.model.eval()
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            return predicted