import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.in_resolution = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 1e-3)
        self.momentum = prm.get('momentum', 0.9)
        self.expand_ratio = prm.get('expand_ratio', 1)
        self.dropout = prm.get('dropout', 0.0)
        self.channels = [32, 64, 128, 256]
        self._set_hyper('conv_layout', [1, 1, 1])
        self.use_res_connect = [True, True, True]
        self.stem_in_channels = self.channels[0]
        self.stem_out_channels = self.channels[1]
        self.stem_resolution = self.in_resolution
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_in_channels),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.stem_in_channels, self.stem_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU6(inplace=True),
        )
        self.block1 = Block(self.stem_out_channels, self.stem_out_channels, expand_ratio=self.expand_ratio)
        self.block2 = Block(self.stem_out_channels, self.stem_out_channels, expand_ratio=self.expand_ratio)
        self.block3 = Block(self.stem_out_channels, self.stem_out_channels, expand_ratio=self.expand_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.stem_out_channels, self.num_classes)

    def _set_hyper(self, name, value):
        setattr(self, name, value)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            out = self(x)
            loss = self.criteria[0](out, y)
            loss.backward()
            self.optimizer.step()