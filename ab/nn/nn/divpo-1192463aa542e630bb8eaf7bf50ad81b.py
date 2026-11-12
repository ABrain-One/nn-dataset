import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.eps = 1e-3

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
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
        self.model = InvertedResidual(32, 32, 2, 1)
        self.input_size = 32
        self.num_classes = out_shape[0]
        self.dropout = prm.get('dropout', 0.1)
        self.channels = [32, 32, 32]
        self.conv_params = [
            (32, 3, 1, 1, 2),
            (32, 3, 1, 1, 1),
        ]
        self.block_list = ['InvertedResidual']
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer1 = self.build_block(32, 32, 1, 1, 1)
        self.layer2 = self.build_block(32, 32, 1, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

    def build_block(self, in_channels, out_channels, block_size, stride, expand_ratio):
        block = InvertedResidual(in_channels, out_channels, stride, expand_ratio)
        return block

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.01,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
        )

    def learn(self, data_roll):
        self.train()
        for batch, labels in data_roll:
            batch, labels = batch.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            out = self(batch)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()