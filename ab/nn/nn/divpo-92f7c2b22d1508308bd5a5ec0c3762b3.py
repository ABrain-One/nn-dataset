import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = Block(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.block3 = Block(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block4 = Block(128, 256, kernel_size=3, stride=1, padding=1)
        self.block5 = Block(256, 256, kernel_size=3, stride=1, padding=1)
        self.block6 = Block(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block7 = Block(256, 512, kernel_size=3, stride=1, padding=1)
        self.block8 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.block9 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool4(x)
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
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Auto-generated single-channel detection (used when in_shape[0] == 1)
def is_single_channel(prm):
    return 'single_channel' in prm and prm['single_channel']

def supported_hyperparameters():
    return {'lr', 'momentum', 'single_channel'}

def train_setup(self, prm):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.net = Net(in_shape=(1, 32, 32), out_shape=(10,), prm=prm, device=self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.net.parameters(), lr=prm['lr'], momentum=prm['momentum'])

def learn(self, train_data):
    self.net.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

def validate(self, val_data):
    self.net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def train(self, train_data, val_data, prm):
    self.train_setup(prm)
    for epoch in range(prm['epochs']):
        self.learn(train_data)
        acc = self.validate(val_data)
        print(f'Epoch {epoch + 1}/{prm["epochs"]}, Accuracy: {acc * 100:.2f}%')