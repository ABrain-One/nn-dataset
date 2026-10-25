import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # This is the correct way to derive the input channels
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def __str__(self):
        return "Net[epoch=?, loss=?, accuracy=?]"

def train_setup(self, prm):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=self.device)
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=prm.get('lr', 0.01),
        momentum=prm.get('momentum', 0.9)
    )
    self.criterion = nn.CrossEntropyLoss()

def learn(self, train_data):
    self.model.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

def validate(self, val_data):
    self.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def train(self, prm, train_data, val_data):
    self.train_setup(prm)
    for epoch in range(prm.get('epochs', 10)):
        self.learn(train_data)
        accuracy = self.validate(val_data)
        print(f'Epoch {epoch+1}: loss={self.criterion.loss:.4f}, accuracy={accuracy:.4f}')
    self._model = self.model