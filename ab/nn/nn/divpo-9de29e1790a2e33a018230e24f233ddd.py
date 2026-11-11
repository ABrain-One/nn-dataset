import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
        ]

        # Use nn.Identity() for empty layers to keep shapes the same
        layers += [
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio,
                      kernel_size=3, stride=stride, padding=1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
        ]

        layers += [
            nn.Conv2d(in_channels * expand_ratio, out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

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
        self.classes = out_shape[0]
        self.kernel_size = 3
        self.padding = self.kernel_size // 2
        self.inverted_residual = InvertedResidual(self.in_channels, 16, stride=1, expand_ratio=1)
        self.in_channels = 16
        self.dropout = nn.Dropout(p=0.0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, self.classes)
        self.to(device)

        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

    def forward(self, x):
        x = self.inverted_residual(x)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.max_pool_2(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
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
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.criteria[0](out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()