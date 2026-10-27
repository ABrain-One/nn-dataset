import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.stride = stride

        mid_channels = in_channels * expansion_factor
        self.use_res_conn = self.stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU6(inplace=True),
                )
            )

        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_conn:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.block1 = InvertedResidual(32, 32, stride=1, expansion_factor=3)
        self.block2 = InvertedResidual(32, 32, stride=1, expansion_factor=3)
        self.block3 = InvertedResidual(32, 64, stride=2, expansion_factor=3)
        self.block4 = InvertedResidual(64, 64, stride=1, expansion_factor=3)
        self.block5 = InvertedResidual(64, 64, stride=1, expansion_factor=3)
        self.block6 = InvertedResidual(64, 96, stride=2, expansion_factor=3)
        self.block7 = InvertedResidual(96, 96, stride=1, expansion_factor=3)
        self.block8 = InvertedResidual(96, 96, stride=1, expansion_factor=3)
        self.block9 = InvertedResidual(96, 160, stride=2, expansion_factor=3)
        self.block10 = InvertedResidual(160, 160, stride=1, expansion_factor=3)
        self.block11 = InvertedResidual(160, 160, stride=1, expansion_factor=3)

        self.conv = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(160),
            nn.ReLU6(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(160, num_classes)

        self.to(device); self.criteria = nn.CrossEntropyLoss().to(device);
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device); self.criteria = nn.CrossEntropyLoss().to(self.device);
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train();
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad();
            outputs = self(inputs);
            loss = self.criteria(outputs, labels);
            loss.backward();
            nn.utils.clip_grad_norm_(self.parameters(), 3);
            self.optimizer.step();