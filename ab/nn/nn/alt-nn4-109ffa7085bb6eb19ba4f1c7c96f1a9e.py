import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels = [64, 128, 256, 512]
        self.dropout_rate = prm['dropout']
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        
                     
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[1], self.channels[0], kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
                 
        self.stage1 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
                 
        self.stage2 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
                 
        self.stage3 = nn.Sequential(
            nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
                    
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.channels[3] * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
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