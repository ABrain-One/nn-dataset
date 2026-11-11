import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False) if self.use_res_connect else None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.conv4 is not None:
            identity = self.conv4(identity)

        if self.use_res_connect:
            x = x + identity
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.inverted_residual = InvertedResidual(32, 32, 1, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])
        self.apply_hyperparameters(prm, 32)

    def apply_hyperparameters(self, prm, channels):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        dropout = prm.get('dropout', 0.0)
        weight_decay = prm.get('weight_decay', 0.0)
        epochs = prm.get('epochs', 1)
        batch_size = prm.get('batch_size', 64)
        warmup_epochs = 2
        warmup_lr = 1e-5
        warmup_factor = warmup_epochs / epochs
        warmup_schedule = [
            (warmup_factor * epochs) // 3,
            (warmup_factor * epochs) * 2 // 3,
            warmup_factor * epochs
        ]
        prm['lr'] = warmup_lr
        prm['epochs'] = warmup_epochs
        prm['batch_size'] = batch_size
        prm['schedule'] = warmup_schedule

        opt_args = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer = torch.optim.SGD(self.parameters(), **opt_args)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.inverted_residual(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self(x), y)
            loss.backward()
            self.optimizer.step()