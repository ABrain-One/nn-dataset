import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Ensure this aligns with the input channels of the image data
        num_classes = int(out_shape[0])
        self.conv1 = Block(in_channels, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Block(32, 64, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Block(64, 128, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __str__(self):
        return "CIFAR10_Net"

def train_setup(self, prm):
    self.model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=self.device)
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

def learn(self, train_data):
    self.model.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        self.optimizer.step()

def __repr__(self):
    return "CIFAR10Trainer(in_shape=(3, 32, 32), out_shape=(10,), prm={}, device={})".format(self.prm, self.device)

class CIFAR10Trainer(object):
    def __init__(self, in_shape, out_shape, prm, device):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        self._build_model()

    def _build_model(self):
        train_setup(self, prm=self.prm)

    def train_model(self, train_data):
        self.model = Net(in_shape=self.in_shape, out_shape=self.out_shape, prm=self.prm, device=self.device)
        learn(self, train_data)

    def __repr__(self):
        return __repr__()

    def __str__(self):
        return "CIFAR10Trainer(in_shape={}, out_shape={}, prm={}, device={})".format(self.in_shape, self.out_shape, self.prm, self.device)

# Export the Net class and optionally the CIFAR10Trainer (if it makes sense in the context of the design)
# The exported class name should be Net (not CIFAR10Net or similar)

# Choose the Net class definition based on the input and output shapes
# The output shape should be (num_classes,) and the input shape should match the expected input format

# Example (choose one of the two based on the input shape):
# Net = Net
# Net = Net  # This is a placeholder; the actual class definition will be provided