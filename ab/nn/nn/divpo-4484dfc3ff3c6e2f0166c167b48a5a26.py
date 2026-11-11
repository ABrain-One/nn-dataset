import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU6(inplace=True)
        self.skip = (in_channels == out_channels) and (stride == 1)
        self._init_weights()

    def _init_weights(self):
        for l in [self.conv1, self.conv2]:
            nn.init.normal_(l.weight, mean=0.0, std=0.01)
            if l.bias is not None:
                l.bias.data.zero_()
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.skip:
            y = y + x
        return self.act2(y)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.b1 = Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1)
        self.b2 = Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1)
        self.b3 = Block(32, 64, kernel_size=3, padding=1, stride=2, expansion=1)
        self.b4 = Block(64, 64, kernel_size=3, padding=1, stride=1, expansion=1)
        self.b5 = Block(64, 128, kernel_size=3, padding=1, stride=2, expansion=1)
        self.b6 = Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=1)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat_to = 6 * 6 * 128
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_to, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.pool(x)
        x = x.flatten(1, -1)
        return self.classifier(x)

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