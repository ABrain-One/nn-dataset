import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, kernel_size, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            conv_bn(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            nn.Dropout(drop_rate),
            conv_bn(hidden_dim, out_channels, kernel_size=1),
            nn.Dropout(drop_rate)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.drop_rate = 0.2

        self.features = nn.Sequential(
            conv_bn(self.in_channels, 32, kernel_size=3, stride=2),
            conv_bn(32, 64, kernel_size=3, stride=2),
            Block(64, 64, stride=1, expand_ratio=3, drop_rate=self.drop_rate),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(64, 128, stride=2, expand_ratio=4, drop_rate=self.drop_rate),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(128, 256, stride=2, expand_ratio=6, drop_rate=self.drop_rate),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_bn(256, 512, kernel_size=3, stride=2),
            Block(512, 512, stride=2, expand_ratio=8, drop_rate=self.drop_rate),
            nn.AdaptiveAvgPool2d(6)
        )

        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (6, 6))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}


def supported_hyperparameters():
    return {'lr', 'momentum'}