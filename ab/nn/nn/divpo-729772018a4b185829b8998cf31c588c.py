import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.best_effort_block_1 = self.BestEffortBlock(32)
        self.best_effort_block_2 = self.BestEffortBlock(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.best_effort_block_3 = self.BestEffortBlock(64)
        self.best_effort_block_4 = self.BestEffortBlock(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.best_effort_block_5 = self.BestEffortBlock(128)
        self.best_effort_block_6 = self.BestEffortBlock(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.best_effort_block_7 = self.BestEffortBlock(256)
        self.best_effort_block_8 = self.BestEffortBlock(256)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

        self.criteria = None
        self.optimizer = None

    def BestEffortBlock(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.best_effort_block_1(x) + x
        x = self.best_effort_block_2(x) + x

        x = self.conv2(x)
        x = self.best_effort_block_3(x) + x
        x = self.best_effort_block_4(x) + x

        x = self.conv3(x)
        x = self.best_effort_block_5(x) + x
        x = self.best_effort_block_6(x) + x

        x = self.conv4(x)
        x = self.best_effort_block_7(x) + x
        x = self.best_effort_block_8(x) + x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()