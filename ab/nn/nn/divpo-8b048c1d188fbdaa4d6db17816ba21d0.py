import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio != 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential()

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      stride=stride, padding=1, groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if not self.use_res_connect:
            return self.conv3(self.conv2(self.conv1(x)))

        x = self.conv1(x)
        out = self.conv2(x)
        out = out + x
        return self.conv3(out)

class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.depthwise = InvertedResidual(
            in_channels, out_channels, stride, expand_ratio)

    def forward(self, x):
        return self.depthwise(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            EfficientBlock(32, 32, 1, 4),
            nn.MaxPool2d(2, 2),
            EfficientBlock(32, 64, 2, 4),
            nn.MaxPool2d(2, 2),
            EfficientBlock(64, 128, 2, 4),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.features(x)
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
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.criteria[0](out, y)
            loss.backward()
            self.optimizer.step()