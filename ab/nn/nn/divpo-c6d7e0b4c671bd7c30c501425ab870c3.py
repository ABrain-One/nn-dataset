import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6_1 = Block(64, 128)
        self.block6_2 = Block(128, 128)
        self.block7 = Block(128, 128)
        self.block8 = Block(128, 128)
        self.block9 = Block(128, 128)
        self.block10 = Block(128, 128)
        self.block11 = Block(128, 128)
        self.block12 = Block(128, 64)
        self.block13 = Block(64, 64)
        self.block14 = Block(64, 64)
        self.block15_1 = Block(64, 32)
        self.block15_2 = Block(32, 32)
        self.pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6_1(x)
        x = self.block6_2(x)
        x = self.pool(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15_1(x)
        x = self.block15_2(x)
        x = self.pool(x)
        x = self.avg_pool(x)
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

# Helpers below

__P__ = 3  # pre-padding for block ResBlock(3)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.act = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout(0.2)
        self.equal_chs = in_channels == out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        if not self.equal_chs:
            x = nn.functional.pad(x, (__P__, __P__, __P__, __P__), mode='constant', value=0)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.act(x)

def supported_hyperparameters():
    return {'lr', 'momentum'}