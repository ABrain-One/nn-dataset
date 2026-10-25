import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
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
        in_channels = in_shape[1]  # Follow the in_channels derivation rule
        num_classes = out_shape[0]

        self.conv1 = Block(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Block(64, 128, stride=2)
        self.conv3 = Block(128, 256, stride=2)
        self.conv4 = Block(256, 512, stride=2)

        # Flatten and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv3(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv4(x)
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
            self.optimizer.step()

    def __repr__(self):
        return f"Net(in_shape={self.conv1.conv.in_channels}, out_shape={self.classifier.out_features}, device={self.device})"