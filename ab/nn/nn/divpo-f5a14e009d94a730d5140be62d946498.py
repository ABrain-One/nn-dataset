import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 32, groups=32)

        self.syncbn1 = nn.SyncBatchNorm(32, eps=1e-4, momentum=0.1, affine=False)
        self.dwconv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
        self.syncbn2 = nn.SyncBatchNorm(32, eps=1e-4, momentum=0.1, affine=False)

        self.upscale = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )

        self.layer3 = self._make_layer(Block, 32, 32, 2)
        self.layer4 = self._make_layer(Block, 32, 32, 2)

        self.stem2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer5 = Block(64, 64)
        self.layer6 = Block(64, 64, groups=64)

        self.syncbn3 = nn.SyncBatchNorm(64, eps=1e-4, momentum=0.1, affine=False)
        self.dwconv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.syncbn4 = nn.SyncBatchNorm(64, eps=1e-4, momentum=0.1, affine=False)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        self.to(device)

    def _make_layer(self, block, in_planes, out_planes, blocks):
        layers = [block(in_planes, out_planes)]
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.syncbn1(x)
        x = self.dwconv1(x)
        x = self.syncbn2(x)
        x = self.upscale(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.stem2(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.syncbn3(x)
        x = self.dwconv2(x)
        x = self.syncbn4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        out = self.classifier(x)
        return out

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