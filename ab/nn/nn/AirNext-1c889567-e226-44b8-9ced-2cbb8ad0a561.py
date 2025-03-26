
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AirBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2, ratio=3):
        super(AirBlock, self).__init__()
        mid_channels = out_channels // ratio
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.bn3(self.conv3(x))
        x = self.sigmoid(x)
        return x


class AirNeXtUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, bottleneck_width, ratio):
        super(AirNeXtUnit, self).__init__()
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        self.use_air_block = (stride == 1 and mid_channels < 512)

        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if self.use_air_block:
            self.air = AirBlock(in_channels, group_width, groups=cardinality // ratio, ratio=ratio)

        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        if self.resize_identity:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_air_block:
            att = self.air(x)
            att = F.interpolate(att, size=x.shape[2:], mode="bilinear", align_corners=True)  # Ensure att matches x dimensions
        identity = self.identity_conv(x) if self.resize_identity else x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_air_block:
            x = x * att
        x = self.conv3(x)
        x = x + identity
        x = self.activ(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]

        channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]]
        init_block_channels = 64
        cardinality = 32
        bottleneck_width = 4
        ratio = 2

        self.in_size = image_size
        self.num_classes = class_number

        self.features = nn.Sequential(
            nn.Conv2d(channel_number, init_block_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_block_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), AirNeXtUnit(in_channels, out_channels, stride, cardinality, bottleneck_width, ratio))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
        self.output = nn.Linear(in_channels, class_number)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

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
        self.scheduler.step()


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}
