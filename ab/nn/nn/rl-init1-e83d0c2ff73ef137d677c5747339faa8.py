import torch
from torch import nn, optim
from torchvision.models import resnet50

def supported_hyperparameters():
    return {'lr': 0.01, 'momentum': 0.9}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.channel_number = in_shape[0]
        
        self.criteria = nn.CrossEntropyLoss()
        
        resnet = resnet50(pretrained=True)  
        resnet.fc = nn.Linear(resnet.fc.in_features, out_shape[0]) 
        self.model = resnet

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()