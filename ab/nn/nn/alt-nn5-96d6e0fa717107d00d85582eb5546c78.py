import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.c1 = nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=3, padding=2)
        self.b1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.b2 = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)
        
        self.c4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)
        
        self.c5 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)
        self.m3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(192 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.m1(x)
        
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = self.m2(x)
        
        x = self.c3(x)
        x = self.b3(x)
        x = self.relu(x)
        
        x = self.c4(x)
        x = self.b4(x)
        x = self.relu(x)
        
        x = self.c5(x)
        x = self.b5(x)
        x = self.relu(x)
        x = self.m3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()