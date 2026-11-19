import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_p=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(drop_p) if drop_p > 0.0 else nn.Identity()
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.drop(out)
        return F.gelu(out + identity)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.drop_p = 0.1

        # Use the fixed lr and momentum as required
        self.lr = self.learning_rate
        self.mom = self.momentum

        # Efficient ResNet-style architecture
        # CIFAR-10 is 32x32
        # Stem: 3 -> 64
        self.stem = ConvBNAct(self.in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Stage 1: 64 -> 64
        self.layer1 = self._make_layer(64, 64, 2, stride=1)

        # Stage 2: 64 -> 128
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        # Stage 3: 128 -> 256
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Stage 4: 256 -> 512
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, self.num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_ch, out_ch, stride=stride, drop_p=self.drop_p))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1, drop_p=self.drop_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

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