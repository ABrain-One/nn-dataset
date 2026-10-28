import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class C1(nn.Module):

    def __init__(self, c1: int, c2: int, n: int=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return self.m(y) + y

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=True)]
        self.c1_stage1 = C1(c1=32, c2=64, n=1)
        layers.append(self.c1_stage1)
        self.c1_stage2 = C1(c1=64, c2=128, n=2)
        layers.append(self.c1_stage2)
        self.c1_stage3 = C1(c1=128, c2=128, n=2)
        layers.append(self.c1_stage3)
        self.c1_stage4 = C1(c1=128, c2=64, n=1)
        layers.append(self.c1_stage4)
        self.c1_stage5 = C1(c1=64, c2=32, n=1)
        layers.append(self.c1_stage5)
        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.c1_stage1(x)
        x = self.c1_stage2(x)
        x = self.c1_stage3(x)
        x = self.c1_stage4(x)
        x = self.c1_stage5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()