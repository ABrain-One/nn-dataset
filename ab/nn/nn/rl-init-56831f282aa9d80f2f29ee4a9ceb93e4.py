import torch
from torch import nn, optim
from torch.nn import functional as F

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        self.channel_number = 32
        self.class_number = 10

        self.criteria = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(3, self.channel_number, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channel_number)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(self.channel_number * 6 * 6, self.class_number)

        self.dropout = nn.Dropout(p=self.prm['dropout'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(-1, self.channel_number * 6 * 6)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = optim.SGD([{'params': self.conv1.parameters(), 'lr': self.prm['lr'], 'momentum': self.prm['momentum']},
                                    {'params': self.bn1.parameters(), 'lr': self.prm['lr'], 'momentum': self.prm['momentum']},
                                    {'params': self.fc.parameters(), 'lr': self.prm['lr'], 'momentum': self.prm['momentum']}], lr=self.prm['lr'], momentum=self.prm['momentum'])

    def learn(self, train_data):
        self.train()
        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, targets)
            loss.backward()
            self.optimizer.step()