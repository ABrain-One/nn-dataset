import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio != 1:
            self.conv = nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, hidden_dim, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.expand_conv = InvertedResidual(in_channels, out_channels, stride, expand_ratio=4)
        self.bottleneck_conv = InvertedResidual(out_channels, out_channels, 1, expand_ratio=6)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        return x

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
        self.block4 = Block(32, 32, stride=2)
        self.block5 = Block(32, 64, stride=2)
        self.block6 = Block(64, 64, stride=1)
        self.block7 = Block(64, 64, stride=1)
        self.block8 = Block(64, 64, stride=1)
        self.block9 = Block(64, 64, stride=2)
        self.block10 = Block(64, 96, stride=1)
        self.block11 = Block(96, 96, stride=1)
        self.block12 = Block(96, 96, stride=1)
        self.block13 = Block(96, 96, stride=1)
        self.block14 = Block(96, 96, stride=1)
        self.block15 = Block(96, 96, stride=1)
        self.block16 = Block(96, 96, stride=1)
        self.block17 = Block(96, 96, stride=1)
        self.block18 = Block(96, 96, stride=1)
        self.block19 = Block(96, 96, stride=1)
        self.block20 = Block(96, 160, stride=2)
        self.block21 = Block(160, 160, stride=1)
        self.block22 = Block(160, 160, stride=1)
        self.block23 = Block(160, 160, stride=1)
        self.block24 = Block(160, 160, stride=1)
        self.block25 = Block(160, 320, stride=2)

        self.features = nn.Sequential(
            self.stem,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
            self.block7,
            self.block8,
            self.block9,
            self.block10,
            self.block11,
            self.block12,
            self.block13,
            self.block14,
            self.block15,
            self.block16,
            self.block17,
            self.block18,
            self.block19,
            self.block20,
            self.block21,
            self.block22,
            self.block23,
            self.block24,
            self.block25,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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