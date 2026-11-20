import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        # Architecture: 4 convolution stages, up to 256 channels, BatchNorm, Residual connections
        # Stage 1: 3 -> 64
        self.conv1_1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)

        # Stage 2: 64 -> 128 (Downsample)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(128)
        # Residual projection for stage 2
        self.res2_proj = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)

        # Stage 3: 128 -> 256 (Downsample)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(256)
        # Residual projection for stage 3
        self.res3_proj = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)

        # Stage 4: 256 -> 256
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        # Residual projection for stage 4 (identity, but defined for consistency if needed, though here channels match)
        # Since channels match (256->256) and stride is 1, no projection needed for residual add, just identity.

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_shape[0])

    def forward(self, x):
        # Stage 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))

        # Stage 2
        res = self.res2_proj(x)
        out = self.relu(self.bn2_1(self.conv2_1(x)))
        out = self.bn2_2(self.conv2_2(out))
        out += res
        x = self.relu(out)

        # Stage 3
        res = self.res3_proj(x)
        out = self.relu(self.bn3_1(self.conv3_1(x)))
        out = self.bn3_2(self.conv3_2(out))
        out += res
        x = self.relu(out)

        # Stage 4
        res = x
        out = self.relu(self.bn4_1(self.conv4_1(x)))
        out = self.bn4_2(self.conv4_2(out))
        out += res
        x = self.relu(out)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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