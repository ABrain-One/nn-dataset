import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.c1 = nn.Conv2d(in_shape[1], 32, kernel_size=7, stride=3, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c2 = nn.Conv2d(32, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.c4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.c5 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.dropout = nn.Dropout(prm['dropout'])
        
        self.fc1 = nn.Linear(192 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        
        self.fc3 = nn.Linear(4096, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        
        x = self.c2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        
        x = self.c3(x)
        x = self.relu3(x)
        
        x = self.c4(x)
        x = self.relu4(x)
        
        x = self.c5(x)
        x = self.relu5(x)
        x = self.mp5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()