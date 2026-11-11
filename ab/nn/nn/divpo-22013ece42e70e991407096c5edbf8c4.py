import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    # Version of nn.Conv2d that wraps self.conv and self.act with self.bn
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=1)
        self.act = nn.ReLU() if act == 'relu' else nn.GELU()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))

    def fuse(self):
        # Fuse batchnorm and activation
        self.conv.fuse_bias_and_activation()
        self.bn = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.dropout = prm.get('dropout', 0.0)
        self.in_channels = self.in_channels
        self.model_size = 3000000

        self.block1 = Block(self.in_channels, 32, 3)
        self.block2 = Block(32, 32, 3)
        self.block3 = Block(32, 64, 3)
        self.block4 = Block(64, 64, 3)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.block5 = Block(64, 64, 3)
        self.block6 = Block(64, 128, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()