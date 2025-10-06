import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr': 0.001, 'momentum': 0.9, 'dropout': 0.4}

class BagNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bottleneck_factor=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = self.conv1x1_block(in_channels, mid_channels)
        self.conv2 = self.conv_block(mid_channels, mid_channels, kernel_size, stride)
        self.conv3 = self.conv1x1_block(mid_channels, out_channels, activation=False)

    @staticmethod
    def conv1x1_block(in_channels, out_channels, activation=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if activation else nn.Identity(),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride):
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


class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body = BagNetBottleneck(in_channels, out_channels, kernel_size, stride)

        if self.resize_identity:
            self.identity_conv = self.conv1x1_block(in_channels, out_channels, activation=False)

        self.activ = nn.ReLU(inplace=True)

    @staticmethod
    def conv1x1_block(in_channels, out_channels, activation=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if activation else nn.Identity(),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x):
        identity = x
        if self.resize_identity:
            identity = self.identity_conv(x)

        x = self.body(x)

        if x.size(2) != identity.size(2) or x.size(3) != identity.size(3):
            identity = nn.functional.interpolate(identity, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return self.activ(x + identity)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]

                        
        layers += [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        in_channels = 64

                 
        layers += [
            BagNetUnit(in_channels, 192, kernel_size=3, stride=2),
            BagNetUnit(192, 192, kernel_size=3, stride=1),
            BagNetUnit(192, 256, kernel_size=3, stride=2),
        ]
        in_channels = 256

                 
        layers += [
            BagNetUnit(in_channels, 384, kernel_size=3, stride=2),
            BagNetUnit(384, 384, kernel_size=3, stride=1),
            BagNetUnit(384, 384, kernel_size=3, stride=2),
        ]
        in_channels = 384

                 
        layers += [
            BagNetUnit(in_channels, 512, kernel_size=3, stride=2),
            BagNetUnit(512, 512, kernel_size=3, stride=1),
            BagNetUnit(512, 512, kernel_size=3, stride=2),
        ]
        in_channels = 512

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        dropout_p = prm['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(3072, out_shape[0]),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4,
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()