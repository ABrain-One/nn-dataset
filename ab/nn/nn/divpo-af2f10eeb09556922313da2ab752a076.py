import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size, padding=padding, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * expansion)
        self.conv3 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.2)
        self.out_channels = out_channels
        self.stride = stride

        self.res_path = in_channels == out_channels and stride == 1

    def swish(self, x):
        return x * x.sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.drop is not None:
            out = self.drop(out)

        if self.res_path:
            out = out + residual

        return out


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.b1 = Block(32, 32, expansion=1)
        self.b2 = Block(32, 32, expansion=1)
        self.b3 = Block(32, 32, expansion=1)
        self.b4 = Block(32, 64, expansion=4, stride=2)
        self.b5 = Block(64, 64, expansion=1)
        self.b6 = Block(64, 64, expansion=1)
        self.b7 = Block(64, 64, expansion=1)
        self.b8 = Block(64, 128, expansion=4, stride=2)
        self.b9 = Block(128, 128, expansion=1)
        self.b10 = Block(128, 128, expansion=1)
        self.b11 = Block(128, 128, expansion=1)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)
        x = self.b10(x)
        x = self.b11(x)
        x = self.pool(x)
        x = self.flat(x)
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