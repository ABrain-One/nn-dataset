import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_channels(channels):
    return 32 // channels

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_res:
            return x + self.act2(self.bn3(self.conv3(self.act1(self.bn1(self.conv1(x))))))
        else:
            return self.act2(self.bn3(self.conv3(self.act1(self.bn1(self.conv1(x))))))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.channels = in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block1 = InvertedResidual(32, 32, 1, 1)
        self.block2 = InvertedResidual(32, 64, 2, 6)
        self.block3 = InvertedResidual(64, 64, 1, 6)
        self.block4 = InvertedResidual(64, 128, 2, 6)
        self.block5 = InvertedResidual(128, 128, 1, 6)
        self.block6 = InvertedResidual(128, 256, 2, 6)
        self.block7 = InvertedResidual(256, 256, 1, 6)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, self.num_classes)
        self.to(self.device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.01,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()