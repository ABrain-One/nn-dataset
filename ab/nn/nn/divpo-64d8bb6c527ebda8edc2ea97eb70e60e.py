import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drop = nn.Dropout(prm.get('dropout', 0.2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.drop1 = nn.Dropout(prm.get('dropout', 0.2))
        self.conv5 = nn.Conv2d(64, 72, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(72)
        self.conv6 = nn.Conv2d(72, 72, kernel_size=3, stride=1, padding=1)
        self.drop2 = nn.Dropout(prm.get('dropout', 0.2))
        self.adap_avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(7 * 7 * 72, self.num_classes)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x1 = self.b1(x)
        x = self.conv4(x)
        x = self.drop1(x)
        x = self.conv5(x)
        x2 = self.b2(x)
        x = self.conv6(x)
        x = self.drop2(x)
        x = self.adap_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()