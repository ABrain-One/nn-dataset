import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.layer1 = Block(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer4 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
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

# The above code defines the Net class but does not execute it.
# The following code is expected to be part of the hyper-parameter tuning process
# and should generate a valid PyTorch model definition.

# Example input for Net.__init__ to comply with the contract:
in_shape = (3, 32, 32)  # channels-first (C, H, W) shape for CIFAR-10 images
out_shape = (10,)  # target classes for CIFAR-10 classification
prm = {'lr': 0.01, 'momentum': 0.9}  # example hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Expected Net instantiation and return statement:
def export(in_shape, out_shape, prm, device):
    model = Net(in_shape, out_shape, prm, device)
    return model

# The variable 'export' should contain the entire Net class definition,
# including class definition (Net(in_shape, out_shape, prm, device)),
# static methods (if any), and the return statement at the end:
# export = Net(in_shape, out_shape, prm, device)