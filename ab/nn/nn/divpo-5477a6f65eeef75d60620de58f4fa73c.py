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
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

params = 500000

# Automatically generated design with channel count derived from in_shape
in_channels = 3  # Corresponds to the number of input channels for CIFAR-10
hidden_channels = [32, 64, 128]
kernel_sizes = [3, 3, 3]
strides = [1, 2, 2]
padding = [1, 1, 1]

# Ensure the total parameter count is under the limit
total_params = sum(c * k * k for c, k, s, p in zip(hidden_channels, kernel_sizes, strides, padding))
while total_params > params:
    # Remove the largest kernel/stride/padding combination to stay under the parameter limit
    max_index = total_params.index(max(total_params))
    total_params -= hidden_channels[max_index] * kernel_sizes[max_index] * kernel_sizes[max_index] * (strides[max_index] + padding[max_index])
    del hidden_channels[max_index], kernel_sizes[max_index], strides[max_index], padding[max_index]

# Define the network with the pruned configuration
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_channels, hidden_channels[0], kernel_sizes[0], padding=padding[0])
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(Block(hidden_channels[i], hidden_channels[i + 1], kernel_sizes[i + 1], stride=strides[i + 1], padding=padding[i + 1]))
        self.blocks = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(hidden_channels[-1] * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
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
        self.criteria = nn.CrossEntropyLoss().to(self.device)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

def supported_hyperparameters():
    return {'lr', 'momentum'}