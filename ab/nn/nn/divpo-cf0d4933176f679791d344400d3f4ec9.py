import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 4096, bias=False),
            self.relu,
            self.dropout,
            nn.Linear(4096, num_classes, bias=False)
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.avgpool(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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


def get_ballance(model):
    # returns number of input and number of output channels of the last block
    # we assume that the input and output of the last block are the same
    # this is a simplification - real implementation would need to check all blocks
    # and return the maximum number of input and output channels
    return max(block.in_channels for block in model.blocks)


class Block(nn.Module):
    """ Residual block with `in_channels` (C in) => `out_channels` (C out) """

    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2)) if downsample else nn.Identity()
        self.dropout = nn.Dropout(p=0.3)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.conv3(x)
            residual = self.bn3(residual)

        out += residual
        out = self.relu(out)
        return out

    @staticmethod
    def from_channels(in_channels, out_channels, downsample):
        return Block(in_channels, out_channels, downsample)

    @property
    def in_channels(self):
        return self._in_channels

    @in_channels.setter
    def in_channels(self, channels):
        self._in_channels = channels
        self.conv1.in_channels = channels
        self.bn1.num_features = channels
        if self.downsample:
            self.conv3.in_channels = channels
            self.bn3.num_features = channels


class TinyNet(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.image_size = 32
        self.num_classes = out_shape[0]
        self.blocks = self._build_blocks()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )

    def _build_blocks(self):
        blocks = []
        in_channels = self.in_channels
        out_channels = 32

        for i in range(3):
            downsample = i > 0
            block = Block(in_channels, out_channels, downsample)
            block.in_channels = in_channels
            blocks.append(block)
            in_channels = block.out_channels

        return blocks

    def _forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        return self.avgpool(x)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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


def supported_hyperparameters():
    return {'lr', 'momentum'}