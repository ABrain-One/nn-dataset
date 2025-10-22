import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256]]
        self.in_size = in_shape[1]
        self.num_classes = out_shape[0]
        self.dropout = prm['dropout']

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = 64
        for i, stage_channels in enumerate(self.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(stage_channels):
                stride = 2 if (j == 0 and i > 0) else 1
                stage.add_module(f"unit{j + 1}", BagNetUnit(in_channels, out_channels, kernel_size=3, stride=stride))
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d((6, 6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
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

class BagNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = self.conv1x1_block(in_channels, out_channels)
        self.conv2 = self.conv_block(out_channels, out_channels, kernel_size, stride)
        self.conv3 = self.conv1x1_block(out_channels, out_channels, activation=False)

    def conv1x1_block(self, in_channels, out_channels, activation=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x