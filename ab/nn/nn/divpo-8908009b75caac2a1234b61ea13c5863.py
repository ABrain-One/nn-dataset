import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio != 1:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(Block, self).__init__()
        self.conv = InvertedResidual(in_channels, out_channels, stride, expand_ratio)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        layers = []

        # EfficientNet-like backbone
        # This is a simplified version of the EfficientNet architecture, inspired by MobileNetV3.
        # The inverted residual block (InvertedResidual) is inspired by MobileNetV3's inverted bottlenecks.
        # The block (Block) is an additional wrapper around InvertedResidual with BatchNorm and ReLU.
        # EfficientNet-like blocks are designed to maximize depth-to-width ratio.
        # The block structure is designed to be as similar to the original EfficientNet blocks as possible,
        # while minimizing the number of blocks to stay under the parameter limit.
        # The block layout is designed to maximize depth (block stacking) and minimize width (channel count).
        # This layout aims to emulate the depth-first block stacking of EfficientNet-like architectures.

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        layers.append(Block(32, 32, stride=1, expand_ratio=3))

        layers += [
            # Layer 1
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 2
            Block(32, 32, stride=1, expand_ratio=3),
            # Layer 3
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 4
            Block(32, 32, stride=1, expand_ratio=3),
            # Layer 5
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 6
            Block(32, 32, stride=1, expand_ratio=3),
            # Layer 7
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 8
            Block(32, 32, stride=1, expand_ratio=3),
            # Layer 9
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 10
            Block(32, 32, stride=1, expand_ratio=3),
            # Layer 11
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 12
            Block(32, 32, stride=1, expand_ratio=3),
            # Layer 13
            Block(32, 32, stride=1, expand_ratio=4),
            # Layer 14
            Block(32, 32, stride=1, expand_ratio=3),
        ]

        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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