import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # t channel
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio, momentum=0.99, eps=1e-3),
            nn.ReLU6(inplace=True),
            # k+1 channel
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio, momentum=0.99, eps=1e-3),
            nn.ReLU6(inplace=True),
            # p channel
            nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x_ = x.clone()
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.proj2(x)
        return x_ + x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 64, 1, 2),
            AttentionBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(64, 96, 1, 2),
            InvertedResidual(96, 192, 1, 2),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten()
        )
        num_classes = out_shape[0]
        self.classifier = nn.Linear(192 * 7 * 7, num_classes)
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
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
            out = self.forward(x)
            loss = self.criteria[0](out, y)
            loss.backward()
            self.optimizer.step()