import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
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
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(32, 32, 1, 3.67),
            InvertedResidual(32, 64, 2, 3.67),
            InvertedResidual(64, 64, 1, 3.67),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(64, 128, 2, 3.67),
            InvertedResidual(128, 128, 1, 3.67),
            InvertedResidual(128, 256, 2, 3.67),
            InvertedResidual(256, 256, 1, 3.67),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(256, 512, 2, 3.67),
            InvertedResidual(512, 512, 1, 3.67),
            InvertedResidual(512, 512, 1, 3.67),
            InvertedResidual(512, 512, 1, 3.67),
            InvertedResidual(512, 512, 1, 3.67),
            InvertedResidual(512, 512, 1, 3.67),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Linear(512 * 4 * 4, 10)
        self.to(device); self.train()
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria[0](output, target)
            loss.backward()
            self.optimizer.step()