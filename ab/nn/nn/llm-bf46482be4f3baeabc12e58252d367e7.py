import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = nn.Sequential(nn.Conv2d(in_shape[1], 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 256, kernel_size=5, padding=2), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(p=0.4006452414229993), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=False), nn.Dropout(p=0.24020712119049517), nn.Linear(4096, 4096), nn.ReLU(inplace=False), nn.Linear(4096, out_shape[0]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.0001875888848978332)

    def learn(self, data_roll):
        self.train()
        for data, target in data_roll:
            data, target = (data.to(self.device), target.to(self.device))
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()