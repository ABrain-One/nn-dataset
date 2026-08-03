import torch
import torch.nn as nn

class SEBlock(nn.Module):

    def __init__(self, in_c, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_c, in_c // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_c // reduction, in_c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)

class DoubleConvSE(nn.Module):

    def __init__(self, in_c, out_c):
        super(DoubleConvSE, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.relu(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.feature_extractor = self._build_feature_extractor()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_shape[0])

    def _build_feature_extractor(self):
        model = nn.Sequential()
        model.add_module('dcse1', DoubleConvSE(3, 32))
        model.add_module('mp1', nn.MaxPool2d(2))
        model.add_module('dcse2', DoubleConvSE(32, 64))
        model.add_module('mp2', nn.MaxPool2d(2))
        model.add_module('dcse3', DoubleConvSE(64, 128))
        model.add_module('mp3', nn.MaxPool2d(2))
        return model

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=0.0001)

    def learn(self, data_loader):
        self.train()
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()