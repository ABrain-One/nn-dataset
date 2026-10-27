import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Pointwise
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        drop_rate = prm.get('dropout', 0.2)
        self.momentum = prm.get('momentum', 0.9)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
        )

        self.block1 = Block(32, 32, prm, self.momentum)
        self.block2 = Block(32, 32, prm, self.momentum)
        self.block3 = Block(32, 64, prm, self.momentum)
        self.block4 = Block(64, 64, prm, self.momentum)
        self.block5 = Block(64, 96, prm, self.momentum)
        self.block6 = Block(96, 96, prm, self.momentum)
        self.block7 = Block(96, 96, prm, self.momentum)
        self.block8 = Block(96, 96, prm, self.momentum)
        self.block9 = Block(96, 96, prm, self.momentum)
        self.block10 = Block(96, 96, prm, self.momentum)
        self.block11 = Block(96, 192, prm, self.momentum)

        self.conv = nn.Sequential(
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU6(inplace=True),
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
        )

        self.dropout = nn.Dropout(drop_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x + x)
        x = self.block5(x + x)
        x = self.block6(x + x)
        x = self.block7(x + x)
        x = self.block8(x + x)
        x = self.block9(x + x)
        x = self.block10(x + x)
        x = self.block11(x + x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, prm, momentum):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)