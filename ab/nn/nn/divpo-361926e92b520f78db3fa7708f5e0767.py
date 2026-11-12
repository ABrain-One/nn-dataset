import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se):
        super().__init__()
        self.use_se = use_se
        hidden_dim = in_channels * expand_ratio
        self.kernel_size = 3
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        if stride == 1 and in_channels == hidden_dim:
            self.skip = True
        else:
            self.skip = False

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=self.kernel_size, stride=stride, padding=self.padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = nn.Sequential()
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU6(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )

        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.skip:
            residual = x
        else:
            residual = x[0] if isinstance(x, tuple) else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.use_se:
            x = self.se(x)

        if self.skip:
            return x + residual
        else:
            return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 4, False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(64, 64, 1, 4, False),
        )
        self.classifier = nn.Linear(64, out_shape[0])
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()