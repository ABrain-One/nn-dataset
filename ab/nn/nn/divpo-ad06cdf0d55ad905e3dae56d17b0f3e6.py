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
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(Block, self).__init__()
        self.stride = stride

        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion)
        self.relu = nn.ReLU6(inplace=True)

        self.depthwise_conv = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion)
        self.bn2 = nn.BatchNorm2d(in_channels * expansion)

        self.pointwise_conv2 = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.pointwise_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.pointwise_conv2(out)
        out = self.bn3(out)

        if self.stride != 1 or x.shape[1] != out.shape[1]:
            x = self.skip(x)

        return self.relu(x + out)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.in_channels = 32
        self.num_blocks = [3, 7, 3]
        self.expansion = 1

        self.stem = conv_bn(in_channels, self.in_channels, kernel_size=3, stride=2)

        self.layer1 = self.make_layer(self.in_channels, self.in_channels * 2, stride=2, blocks=self.num_blocks[0])
        self.layer2 = self.make_layer(self.in_channels * 2, self.in_channels * 4, stride=2, blocks=self.num_blocks[1])
        self.layer3 = self.make_layer(self.in_channels * 4, self.in_channels * 8, stride=2, blocks=self.num_blocks[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.in_channels * 8, num_classes)

    def make_layer(self, in_channels, out_channels, stride, blocks):
        layers = [Block(in_channels, out_channels, stride=stride, expansion=self.expansion)]
        for _ in range(blocks - 1):
            layers.append(Block(out_channels, out_channels, stride=1, expansion=self.expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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