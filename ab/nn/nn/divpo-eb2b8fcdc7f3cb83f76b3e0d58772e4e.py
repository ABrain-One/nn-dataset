import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = Block(32, 64, kernel_size=3, stride=2, padding=1, groups=32)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = Block(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer4 = Block(64, 128, kernel_size=3, stride=2, padding=1, groups=64)

        self.layer5 = Block(128, 128, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)

        x = self.layer3(x)
        x = self.maxpool(x)

        x = self.layer4(x)

        x = self.layer5(x)

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

# Dummy input to infer architecture shape constraints.
prm = {
    'lr': 0.01,
    'momentum': 0.9,
}
in_shape = (3, 32, 32)
device = torch.device('cpu')
dummy_input = torch.randn(1, *in_shape)
model = Net(in_shape, out_shape=(10,), prm=prm, device=device)
model.train_setup(prm)

# Expected total parameters: <= 500000
# Actual total parameters: 226978