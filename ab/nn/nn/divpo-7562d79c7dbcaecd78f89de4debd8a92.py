import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.features, self.features, kernel_size=3,
                               stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.features2 = 64
        self.conv3 = nn.Conv2d(self.features, self.features2, kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.features2, self.features2, kernel_size=3,
                               stride=1, padding=1)
        self.drop2 = nn.Dropout(0.25)
        self.features3 = 128
        self.conv5 = nn.Conv2d(self.features2, self.features3, kernel_size=3,
                               stride=1, padding=1)
        self.conv6 = nn.Conv2d(self.features3, self.features3, kernel_size=3,
                               stride=1, padding=1)
        self.drop3 = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(self.features3)
        self.relu = nn.ReLU(inplace=True)
        self.n = self.image_size // 4
        self.avgpool = nn.AdaptiveAvgPool2d((self.n, self.n))
        self.fc = nn.Linear(self.features3 * self.n * self.n, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.drop3(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm: dict) -> None:
        """Setup for training."""
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate,
                                          momentum=self.momentum, weight_decay=1e-4)

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        """Train the model on the provided data."""
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            self.optimizer.step()