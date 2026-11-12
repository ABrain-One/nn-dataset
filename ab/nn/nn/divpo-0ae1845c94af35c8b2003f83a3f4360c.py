import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.features, self.features * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.features * 2, self.features * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(self.features * 4, self.features * 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.features * 8, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(nn.functional.relu(x))
        x = self.conv2(x)
        x = self.pool(nn.functional.relu(x))
        x = self.conv3(x)
        x = self.pool(nn.functional.relu(x))
        x = self.conv4(x)
        x = self.pool(nn.functional.relu(x))
        x = self.drop(x)
        x = self.avgpool(x)
        x = nn.functional.relu(torch.flatten(x, 1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()