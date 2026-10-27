import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_shortcut = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0),
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        if self.use_shortcut:
            return x + self.block(x)
        else:
            return self.block(x)

# Net definition
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = ConvBNReLU(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = InvertedResidual(32, 32, stride=1, expand_ratio=1)
        self.layer2 = InvertedResidual(32, 64, stride=2, expand_ratio=6)
        self.layer3 = InvertedResidual(64, 96, stride=2, expand_ratio=6)
        self.layer4 = InvertedResidual(96, 160, stride=2, expand_ratio=6)
        self.layer5 = InvertedResidual(160, 320, stride=2, expand_ratio=6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(320, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()