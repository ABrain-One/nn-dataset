import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
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
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # Follow the guideline for channel derivation
        self.num_classes = out_shape[0]
        self.conv1 = Block(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

        self.optimizer = None
        self.criteria = None

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, num_classes={self.num_classes})"

    def __str__(self):
        return self.__repr__()

    def __del__(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            del self.optimizer
        if self.criteria is not None:
            del self.criteria
        if hasattr(self, 'conv1') and hasattr(self.conv1, 'weight'):
            del self.conv1.weight
        if hasattr(self, 'conv2') and hasattr(self.conv2, 'weight'):
            del self.conv2.weight
        if hasattr(self, 'conv3') and hasattr(self.conv3, 'weight'):
            del self.conv3.weight
        if hasattr(self, 'classifier') and hasattr(self.classifier, 'weight'):
            del self.classifier.weight