import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        # Stage 1: 3 -> 64
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 2: 64 -> 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Stage 3: 128 -> 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # Residual-style skip connection in deeper feature stages (Stage 3)
        # Since input (128) != output (256), we use a 1x1 conv to match dimensions
        self.skip_proj = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)

        # Classifier head
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.max_pool2d(x, 2)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.max_pool2d(x, 2)

        # Stage 3 (with residual skip)
        identity = self.skip_proj(x)
        
        out = self.bn3(self.conv3(x))
        out = F.relu(out + identity, inplace=True)
        
        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

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