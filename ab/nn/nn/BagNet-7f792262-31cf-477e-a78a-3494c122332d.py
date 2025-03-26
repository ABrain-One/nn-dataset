
import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class BagNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bottleneck_factor=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_factor

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


class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.body = BagNetBottleneck(in_channels, out_channels, kernel_size, stride)

        if self.resize_identity:
            self.identity_conv = self.conv1x1_block(in_channels, out_channels, activation=False)
        self.activ = nn.ReLU(inplace=True)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

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
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        self.channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]]
        self.in_size = image_size
        self.num_classes = class_number

        self.features = nn.Sequential(
            nn.Conv2d(channel_number, 64, kernel_size=5, stride=3, padding=2),  # Changed kernel_size from 7 to 5
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=4, padding=0),  # Changed padding from 1 to 0
        )

        in_channels = 64
        for i, stage_channels in enumerate(self.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(stage_channels):
                stride = 2 if (j == 0 and i > 0) else 1
                stage.add_module(f"unit{j + 1}", BagNetUnit(in_channels, out_channels, kernel_size=3, stride=stride))
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
        self.output = nn.Linear(in_channels, self.num_classes)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.output(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'],)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
