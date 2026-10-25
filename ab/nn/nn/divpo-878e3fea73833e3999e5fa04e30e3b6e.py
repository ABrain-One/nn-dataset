import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.residual = stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expansion_factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expanded_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            return x + self.block(x)
        else:
            return self.block(x)

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

        self.block1 = InvertedResidual(32, 32, 1, 3)
        self.block2 = InvertedResidual(32, 32, 1, 3)
        self.block3 = InvertedResidual(32, 64, 2, 3)
        self.block4 = InvertedResidual(64, 64, 1, 3)
        self.block5 = InvertedResidual(64, 96, 1, 3)
        self.block6 = InvertedResidual(96, 96, 1, 3)
        self.block7 = InvertedResidual(96, 96, 1, 3)
        self.block8 = InvertedResidual(96, 96, 2, 3)
        self.block9 = InvertedResidual(96, 96, 1, 3)
        self.block10 = InvertedResidual(96, 192, 1, 3)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, num_classes),
        )

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
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device); self.criteria = nn.CrossEntropyLoss().to(self.device)
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