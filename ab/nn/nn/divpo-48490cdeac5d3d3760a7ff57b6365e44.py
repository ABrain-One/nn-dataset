import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Standard utility class to provide same padding on Conv2d
class SamePadding2d:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, input_size):
        pad_h = max((self.stride[0] - self.kernel_size[0]) // 2 + (self.kernel_size[0] - 1) // self.stride[0], 0)
        pad_w = max((self.stride[1] - self.kernel_size[1]) // 2 + (self.kernel_size[1] - 1) // self.stride[1], 0)
        return (pad_h, pad_w)

# Block definition
class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Main class
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.in_resolution = in_shape[2]
        out_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 1e-3)
        self.momentum = prm.get('momentum', 0.9)
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128),
            nn.AdaptiveAvgPool2d(6)
        )
        self.classifier = nn.Linear(128 * 6 * 6, out_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 1e-3),
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()