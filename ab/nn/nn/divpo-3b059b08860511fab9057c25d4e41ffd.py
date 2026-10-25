import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Pointwise
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if not self.use_res_connect:
            return self.conv(x)
        else:
            return x + self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.block1 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.block2 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.block3 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.block4 = InvertedResidual(32, 64, stride=2, expand_ratio=3)
        self.block5 = InvertedResidual(64, 64, stride=1, expand_ratio=3)
        self.block6 = InvertedResidual(64, 64, stride=1, expand_ratio=3)
        self.block7 = InvertedResidual(64, 64, stride=1, expand_ratio=3)
        self.block8 = InvertedResidual(64, 64, stride=2, expand_ratio=3)
        self.block9 = InvertedResidual(64, 96, stride=1, expand_ratio=3)
        self.block10 = InvertedResidual(96, 96, stride=1, expand_ratio=3)
        self.block11 = InvertedResidual(96, 96, stride=1, expand_ratio=3)
        self.block12 = InvertedResidual(96, 192, stride=2, expand_ratio=3)

        self.features = nn.Sequential(
            self.stem,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
            self.block7,
            self.block8,
            self.block9,
            self.block10,
            self.block11,
            self.block12,
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Linear(192 * 6 * 6, num_classes)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()