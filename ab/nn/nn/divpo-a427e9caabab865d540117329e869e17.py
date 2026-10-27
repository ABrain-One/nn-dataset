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
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.conv1 = Block(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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
            self.optimizer.step()

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.conv1) + ')'

def learn_prm():
    return {'lr': 0.01, 'momentum': 0.9}

def train_setup(net, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.train_setup(prm)
    net.to(device)
    return net, device

def validate(net, data_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(net.device), labels.to(net.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train(net, train_data_loader, val_data_loader, prm):
    device, = train_setup(net, prm)
    for epoch in range(prm['epochs']):
        net.learn(train_data_loader)
        accuracy = validate(net, val_data_loader)
        print(f'Epoch [{epoch+1}/{prm["epochs"]}] - Accuracy: {accuracy}')
    return net

def get_model(in_shape, out_shape, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(in_shape, out_shape, prm, device)
    return train(net, prm['train_data'], prm['val_data'], prm)