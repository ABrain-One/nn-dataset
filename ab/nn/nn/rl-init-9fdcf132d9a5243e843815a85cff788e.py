import torch
from torch import nn

class ProvidedBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        self.learning_rate = prm['lr']
        self.batch_size = prm['batch']
        self.dropout_value = prm['dropout']
        self.momentum = prm['momentum']
        self.transform = prm['transform']

        self.criteria = nn.CrossEntropyLoss()

        self.features = nn.Sequential(
            ProvidedBlock(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ProvidedBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ProvidedBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_value),
            nn.Linear(64 * 7 * 7, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.out_shape[0]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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