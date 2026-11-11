import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.channels = 32
        self.kernel_size = 3
        self.padding = 1
        self.dropout = 0.3
        self.pooling_size = 4
        self.classes = 10
        self.num_classes = int(out_shape[0])

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self.make_layer(self.channels, 1, 32, 1, 1)
        self.layer2 = self.make_layer(32, 1, 64, 2, 2)
        self.layer3 = self.make_layer(64, 1, 128, 2, 2)
        self.layer4 = self.make_layer(128, 1, 256, 2, 2)
        self.layer5 = self.make_layer(256, 1, 512, 2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((self.pooling_size, self.pooling_size))
        self.dropout = nn.Dropout(p=self.dropout)
        self.classifier = nn.Linear(512 * self.pooling_size * self.pooling_size, self.num_classes)

    def make_layer(self, in_channels, block_num, out_channels, stride, repeat_times):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride))
        for i in range(1, repeat_times):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (
            nn.CrossEntropyLoss().to(self.device),
        )
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.02,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9
        )

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out