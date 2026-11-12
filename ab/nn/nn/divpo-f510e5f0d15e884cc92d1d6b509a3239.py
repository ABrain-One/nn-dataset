import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU6(inplace=True)
        self.skip = (in_channels == out_channels) and (stride == 1)
        self.expansion = expansion

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        if self.skip:
            if self.expansion == 1:
                return z
            x = self.conv1(x)
            x = self.bn1(x)
            return x + z
        return z

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.layer1 = Block(32, 32, expansion=1)
        self.layer2 = Block(32, 32, expansion=1)
        self.layer3 = Block(32, 64, kernel_size=3, padding=1, stride=2, expansion=1)
        self.layer4 = Block(64, 64, expansion=1)
        self.layer5 = Block(64, 128, kernel_size=3, padding=1, stride=2, expansion=1)
        self.layer6 = Block(128, 128, expansion=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer5(x)
        x = self.layer6(x)
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