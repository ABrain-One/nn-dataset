import torch
from torch import nn

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']

        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M'])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
                            nn.Dropout(p=self.dropout, inplace=True),
                            nn.Linear(6*6*128, 10)
                        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
    def train_setup(self, prm):
        self.to(self.device)
                                                    
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    def learn(self, train_loader):
        self.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        print('Epoch Finished. Loss: ', running_loss / len(train_loader))