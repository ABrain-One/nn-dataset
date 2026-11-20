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
        
        # CIFAR-10 specific normalization statistics
        self.register_buffer('mu', torch.tensor([0.4914, 0.4822, 0.4465]))
        self.register_buffer('sigma', torch.tensor([0.2470, 0.2435, 0.2616]))

        in_channels = in_shape[1]
        num_classes = out_shape[0]

        # Stage 1: 3x3 conv, 64 channels, stride 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 2: 3x3 conv, 128 channels, stride 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Stage 3: 3x3 conv, 256 channels, stride 2 (with residual skip)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # 1x1 projection to match dimensions for residual connection
        self.conv_proj = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        
        # Stage 4: 3x3 conv, 256 channels, stride 1 (with residual skip)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply normalization
        x = (x - self.mu.view(1, 3, 1, 1)) / self.sigma.view(1, 3, 1, 1)

        # Stage 1
        x = nn.functional.relu(self.bn1(self.conv1(x)), inplace=True)

        # Stage 2
        x = nn.functional.relu(self.bn2(self.conv2(x)), inplace=True)

        # Stage 3 (Residual)
        identity = self.conv_proj(x)
        out = self.conv3(x)
        out = self.bn3(out)
        out += identity
        x = nn.functional.relu(out, inplace=True)

        # Stage 4 (Residual)
        identity = x
        out = self.conv4(x)
        out = self.bn4(out)
        out += identity
        x = nn.functional.relu(out, inplace=True)

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
            inputs = inputs.float()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()