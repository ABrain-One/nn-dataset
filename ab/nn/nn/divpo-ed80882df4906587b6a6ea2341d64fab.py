import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels * expansion_factor,
                in_channels * expansion_factor,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion_factor,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        expansion_factor = 2
        self.block = InvertedResidual(in_channels, out_channels, stride, expansion_factor)

    def forward(self, x):
        return self.block(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 32, stride=1)
        self.block3 = Block(32, 32, stride=1)
        self.block4 = Block(32, 32, stride=1)
        self.block5 = Block(32, 32, stride=2)
        self.block6 = Block(32, 32, stride=1)
        self.block7 = Block(32, 64, stride=2)
        self.block8 = Block(64, 64, stride=1)
        self.block9 = Block(64, 64, stride=1)
        self.block10 = Block(64, 64, stride=1)
        self.block11 = Block(64, 64, stride=2)
        self.block12 = Block(64, 64, stride=1)
        self.block13 = Block(64, 96, stride=1)
        self.block14 = Block(96, 96, stride=1)
        self.block15 = Block(96, 96, stride=1)
        self.block16 = Block(96, 96, stride=1)
        self.block17 = Block(96, 96, stride=2)
        self.block18 = Block(96, 96, stride=1)
        self.block19 = Block(96, 160, stride=2)
        self.block20 = Block(160, 160, stride=1)
        self.block21 = Block(160, 160, stride=1)
        self.block22 = Block(160, 160, stride=1)
        self.block23 = Block(160, 160, stride=1)
        self.block24 = Block(160, 320, stride=2)
        self.block25 = Block(320, 320, stride=1)
        self.block26 = Block(320, 320, stride=1)

        self.conv = nn.Sequential(
            nn.Conv2d(320, 160, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU6(inplace=True),
            nn.Conv2d(160, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
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
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.block24(x)
        x = self.block25(x)
        x = self.block26(x)
        x = self.conv(x)
        x = x.mean(dim=(2, 3))  # Average pooling
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