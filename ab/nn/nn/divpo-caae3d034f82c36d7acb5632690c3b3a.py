import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ChannelAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1)
        )

    def forward(self, x):
        x_avg = self.pool(x)
        x_scale = self.conv(x_avg)
        return x * x_scale

class Block(nn.Module):
    def __init__(self, dim, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = ChannelAttn(dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.attn(x)
        x = self.conv3(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block = Block(32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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