import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (batch, channels, height, width)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.stem.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.feature_extractor = nn.Sequential(
            Block(64, 64, 3, 1, 1, 32),
            Block(64, 128, 3, 2, 1, 16),
            Block(128, 128, 3, 3, 1, 16),
            Block(128, 256, 3, 2, 1, 8),
            Block(256, 256, 3, 3, 1, 8),
            Block(256, 256, 3, 3, 1, 8)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, group_width):
        super(Block, self).__init__()
        self.use_conv = stride > 1 or in_channels != out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.gn1 = nn.GroupNorm(group_width, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels // group_width, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // group_width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels // group_width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.gn3 = nn.GroupNorm(group_width, out_channels)
        self.shortcut = nn.Sequential()
        if self.use_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gn3(out)
        out += self.shortcut(identity)
        out = self.relu1(out)
        return out