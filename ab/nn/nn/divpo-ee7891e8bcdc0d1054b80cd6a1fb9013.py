import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, drop_rate=0.):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.expand_ratio = expand_ratio

        self.conv1 = nn.Conv2d(in_channels, out_channels * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * expand_ratio)
        self.act1 = nn.ReLU6(inplace=False)
        self.conv2 = nn.Conv2d(out_channels * expand_ratio, out_channels * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=out_channels * expand_ratio, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * expand_ratio)
        self.act2 = nn.ReLU6(inplace=False)
        self.conv3 = nn.Conv2d(out_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(drop_rate, inplace=False)

    def forward(self, x):
        if self.use_res_connect:
            if self.drop_rate:
                identity = self.drop(x)
            else:
                identity = x
        else:
            identity = torch.nn.functional.pad(x, [1, 1, 1, 1], mode='constant', value=0)
            identity = nn.functional.avg_pool2d(identity, kernel_size=3, stride=self.stride, padding=0)
            identity = nn.Conv2d(x.shape[1], self.conv3.weight.shape[0], kernel_size=1, stride=1, padding=0, bias=False)(identity)
            identity = nn.BatchNorm2d(self.conv3.weight.shape[0])(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_rate:
            out = self.drop(out)

        return identity + out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        dropout = prm.get('dropout', 0.0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 32, expand_ratio=3, drop_rate=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 32, expand_ratio=3, drop_rate=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = Block(32, 32, stride=1, expand_ratio=3, drop_rate=dropout)
        self.block2 = Block(32, 32, stride=1, expand_ratio=3, drop_rate=dropout)
        self.block3 = Block(32, 32, stride=1, expand_ratio=3, drop_rate=dropout)
        self.block4 = Block(32, 32, stride=1, expand_ratio=3, drop_rate=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()