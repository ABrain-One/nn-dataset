import torch
from torch import nn, optim
from torchvision.models import densenet

def supported_hyperparameters():
    return ['lr', 'momentum']

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.criteria = nn.CrossEntropyLoss()
        self.in_channel = in_shape[0]
        self.num_classes = out_shape[0]
        self.model = densenet.densenet121(pretrained=False).to(device)
        fc_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(fc_features, self.num_classes).to(device)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_loader):
        self.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()