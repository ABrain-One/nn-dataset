
import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class DPN131(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, num_blocks=4, growth_rate=35):
        super(DPN131, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn_down2 = nn.BatchNorm2d(128)
        self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.down3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.bn_down3 = nn.BatchNorm2d(256)
        self.layer3 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.down4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.bn_down4 = nn.BatchNorm2d(512)
        self.layer4 = nn.Sequential(ResBlock(512), ResBlock(512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.bn_down2(self.down2(x)))
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.bn_down3(self.down3(x)))
        x = self.layer3(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.bn_down4(self.down4(x)))
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        model_class = DPN131
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        self.model = model_class(self.channel_number, self.class_number)

        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=1e-4)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches: break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
