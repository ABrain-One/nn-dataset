import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * expansion)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=out_channels * expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * expansion)
        self.conv3 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_res_connect:
            out = out + residual
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(in_channels=32, out_channels=32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=32, out_channels=32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=32, out_channels=64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=64, out_channels=64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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