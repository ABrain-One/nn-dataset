import torch
from torch import nn
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
        
                         
        self.learning_rate = prm['lr']
        self.batch_size = prm['batch']
        self.dropout_value = prm['dropout']
        self.momentum = prm['momentum']
        self.transform = prm['transform']
        
                       
        self.features = self._make_layers(in_channels=3, out_channels=64)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_value),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_value),
            nn.Linear(4096, self.out_shape[0])
        )
        
                                            
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = None
    
    def _make_layers(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
    
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