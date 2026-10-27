import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:  # channels-first (C, H, W)
            in_channels = in_shape[1]  # should be 3 for CIFAR-10
        else:
            in_channels = in_shape[0]  # should be 3 for CIFAR-10
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.flatten(1)
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
            self.optimizer.step()

    def __parameters__(self):
        for param in self.parameters():
            yield param
    def __services__(self):
        return {p.data_ptr(): p.device for p in self.__parameters__()}
    def __flops__(self, input_shape):
        # This is an estimation and may not be accurate.
        # It assumes that all convolutions have the same parameters.
        # Actual FLOPS could be higher due to batch normalization and non-linearities.
        batch_size = input_shape[0]
        total_flops = 0

        # Conv1
        total_flops += batch_size * 3 * 32 * 32 * 3 * 3 * 3 * 4 / 1e9  # Conv1
        total_flops += batch_size * 32 * 32 * 4 / 1e9  # BatchNorm1

        # Layer1
        total_flops += batch_size * 32 * 32 * 32 * 3 * 3 * 32 * 32 / 1e9  # Conv2D x2
        total_flops += batch_size * 32 * 32 * 32 * 2 / 1e9  # BatchNorm2D x2

        # Layer2
        total_flops += batch_size * 32 * 16 * 16 * 32 * 3 * 16 * 16 / 1e9  # Conv2D x2
        total_flops += batch_size * 16 * 16 * 32 * 2 / 1e9  # BatchNorm2D x2

        # Layer3
        total_flops += batch_size * 128 * 8 * 8 * 128 * 3 * 8 * 8 / 1e9  # Conv2D
        total_flops += batch_size * 8 * 8 * 128 * 1 / 1e9  # BatchNorm2D

        # AvgPool
        total_flops += batch_size * 256 * 6 * 6 * 6 / 1e9  # AvgPool

        # Classifier
        total_flops += batch_size * 256 * 6 * 6 * 4096 * 1 / 1e9  # Linear x3

        return total_flops