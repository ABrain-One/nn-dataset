import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Deriving input channels from in_shape[1]
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
            Block(256, 512, 3, 2, 1, 4),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)

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
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, group_channels):
        super(Block, self).__init__()
        self.with_downsample = stride != 1 or in_channels != out_channels
        if self.with_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.conv1 = nn.Conv2d(in_channels, group_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.depthwise_conv = nn.Conv2d(group_channels, group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=group_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(group_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pointwise_conv = nn.Conv2d(group_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.with_downsample:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x + identity