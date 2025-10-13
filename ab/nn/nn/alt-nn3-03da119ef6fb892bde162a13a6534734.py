import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr': 0.001, 'momentum': 0.9, 'dropout': 0.5}

class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = self.conv1x1_block(in_channels, mid_channels)
        self.conv2 = self.conv_block(mid_channels, mid_channels, kernel_size, stride)
        self.conv3 = self.conv1x1_block(mid_channels, out_channels, activation=False)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels = [[128, 128, 128], [192, 192, 192], [256, 256, 256], [512, 512, 512]]
        self.in_size = in_shape
        self.num_classes = out_shape[0]
        self.dropout = prm['dropout']

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            BagNetUnit(128, 128, 3, 1),
            BagNetUnit(128, 192, 3, 2),
            BagNetUnit(192, 256, 3, 1),
            BagNetUnit(256, 256, 3, 2),
            BagNetUnit(256, 256, 3, 1),
            BagNetUnit(256, 384, 3, 2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(384 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()