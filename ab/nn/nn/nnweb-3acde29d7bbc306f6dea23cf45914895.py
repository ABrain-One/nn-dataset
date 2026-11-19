import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.lr = prm['lr']
        self.momentum = prm['momentum']

        # CIFAR-10 specific normalization (mean/std)
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))

        in_channels = in_shape[1]
        num_classes = out_shape[0]

        # Stage 1: 3x3 conv, 64 channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 2: 3x3 conv, 128 channels, stride 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Residual block for Stage 2
        self.res_block_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.bn_res2 = nn.BatchNorm2d(128)

        # Stage 3: 3x3 conv, 256 channels, stride 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # Residual block for Stage 3
        self.res_block_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.bn_res3 = nn.BatchNorm2d(256)

        # Stage 4: 3x3 conv, 256 channels, stride 2
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        # Residual block for Stage 4
        self.res_block_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.bn_res4 = nn.BatchNorm2d(256)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Head
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / self.std

        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        identity = x
        out = self.res_block_2(x)
        out = self.bn_res2(out)
        x = F.relu(out + identity, inplace=True)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        identity = x
        out = self.res_block_3(x)
        out = self.bn_res3(out)
        x = F.relu(out + identity, inplace=True)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        identity = x
        out = self.res_block_4(x)
        out = self.bn_res4(out)
        x = F.relu(out + identity, inplace=True)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x