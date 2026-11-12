import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.downsample(orig)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.first_bn = nn.BatchNorm2d(32)
        self.first_relu = nn.ReLU(inplace=True)
        self.first_maxpool = nn.MaxPool2d(kernel_size=1)
        self.small1 = SmallBlock(32, 32)
        self.small2 = SmallBlock(32, 64, stride=2)
        self.small3 = SmallBlock(64, 128, stride=2)
        self.small4 = SmallBlock(128, 128)
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(128 * 7 * 7, self.num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_maxpool(x)
        x = self.small1(x)
        x = self.small2(x)
        x = self.small3(x)
        x = self.small4(x)
        x = self.gap(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()