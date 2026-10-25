import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expansion_factor
        self.conv = nn.Sequential(
            # Pointwise convolution
            nn.Conv2d(in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),

            # Depthwise convolution
            nn.Conv2d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),

            # Pointwise convolution
            nn.Conv2d(in_channels=expanded_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        # Hyperparameters
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = InvertedResidual(32, 32, stride=1, expansion_factor=3)
        self.layer2 = InvertedResidual(32, 32, stride=2, expansion_factor=3)
        self.layer3 = InvertedResidual(32, 64, stride=2, expansion_factor=4)
        self.layer4 = InvertedResidual(64, 64, stride=1, expansion_factor=4)
        self.layer5 = InvertedResidual(64, 96, stride=1, expansion_factor=6)
        self.layer6 = InvertedResidual(96, 96, stride=2, expansion_factor=6)
        self.layer7 = InvertedResidual(96, 160, stride=2, expansion_factor=6)
        self.layer8 = InvertedResidual(160, 160, stride=1, expansion_factor=6)
        self.layer9 = InvertedResidual(160, 320, stride=1, expansion_factor=6)

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(320, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
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