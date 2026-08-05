import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1).unsqueeze(-1)

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=1, stride=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expansion = expansion
        self.hidden_dim = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, self.hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride, 1, groups=self.hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.se = SEBlock(self.hidden_dim)
        self.conv3 = nn.Conv2d(self.hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
        x = self.se(x)
        x = F.relu6(self.bn3(self.conv3(x)), inplace=True)
        return x + residual

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.channels = [32, 64, 128, 256]
        self.features = nn.Sequential(nn.Conv2d(in_shape[1], self.channels[0], 3, 2, 1, bias=False), nn.BatchNorm2d(self.channels[0]), nn.ReLU6(inplace=True))
        for i in range(1, len(self.channels)):
            ir = InvertedResidual(self.channels[i - 1], self.channels[i], 2, 2)
            self.features.add_module(f'ir_{i}', ir)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(prm.get('dropout', 0.5))
        self.classifier = nn.Linear(self.channels[-1], out_shape[0])

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9), weight_decay=0.0001)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()