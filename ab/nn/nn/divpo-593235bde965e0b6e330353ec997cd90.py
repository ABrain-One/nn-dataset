import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.expanded_channels = in_channels * expansion
        self.use_residual = self.in_channels == self.out_channels and self.stride == 1

        self.conv1 = nn.Conv2d(in_channels, self.expanded_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.expanded_channels)
        self.act1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(self.expanded_channels, self.expanded_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expanded_channels)

        self.conv3 = nn.Conv2d(self.expanded_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.use_residual:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act1(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_residual:
            return self.act1(out + residual)
        else:
            return out


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.block1 = Block(32, 32, 1)
        self.block2 = Block(32, 32, 1)
        self.block3 = Block(32, 64, 2, 2)
        self.block4 = Block(64, 64, 1)
        self.block5 = Block(64, 64, 1)
        self.block6 = Block(64, 64, 2, 2)
        self.block7 = Block(64, 64, 1)
        self.block8 = Block(64, 64, 1)
        self.block9 = Block(64, 128, 2, 2)
        self.block10 = Block(128, 128, 1)
        self.block11 = Block(128, 128, 1)
        self.block12 = Block(128, 128, 1)
        self.block13 = Block(128, 128, 1)
        self.block14 = Block(128, 128, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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