import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=False),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=False),
            nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
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
        self.training = True
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 64, 2, 6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(64 * 6 * 6, self.num_classes)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.in_channels = 64
        self.num_classes = out_shape[0]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def learn(self, train_data):
        self.train()
        for i, (xb, yb) in enumerate(train_data):
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            pred = self(xb)
            loss = self.criterion(pred, yb)
            loss.backward()
            self.optimizer.step()