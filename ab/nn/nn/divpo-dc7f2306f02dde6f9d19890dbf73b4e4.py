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
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(prm.get('dropout', 0.2))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.drop2 = nn.Dropout(prm.get('dropout', 0.3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drop3 = nn.Dropout(prm.get('dropout', 0.4))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, self.num_classes)
        self._features = []
        self._hook()

        self.to(self.device)

    def _hook(self):
        self.layer_register = self.conv3.register_forward_hook(self.save_outputs)

    def save_outputs(self, module, input, output):
        self._features = output

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self.drop(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        self.drop2(x)
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        self.drop3(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)

    def learn(self, train_loader):
        self.train()
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
        self.scheduler.step()