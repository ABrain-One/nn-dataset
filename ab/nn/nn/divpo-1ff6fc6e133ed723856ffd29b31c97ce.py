import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        if hasattr(self.bn, 'weight') and self.bn.weight is not None:
            fan_out = torch.nn.init.calculate_gain('relu') * self.conv.in_channels
            self.conv.weight.data = self.conv.weight.data.div(self.bn.weight.data.view(-1, 1, 1, 1).clamp(min=1e-5).expand(self.conv.weight.shape))
            self.conv.bias.data = self.conv.bias.data.div(self.bn.weight.data.div(self.bn.bias.data) if self.bn.bias is not None else 1.0)
            del self.bn
            self.inplace = True

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape as per the guide
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(64, 64, stride=1)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 128, stride=1)
        self.block4 = Block(128, 256, stride=2)
        self.block5 = Block(256, 256, stride=1)
        self.block6 = Block(256, 512, stride=2)
        self.block7 = Block(512, 512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
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

    def __repr__(self):
        return f'Net(in_shape={self.in_shape}, out_shape={self.out_shape}, prm={self.prm}, device={self.device})'