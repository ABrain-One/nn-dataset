import torch
from torch import nn

def supported_hyperparameters():
    return ['lr', 'momentum', 'batch', 'transform', 'epoch']

class Net(nn.Module):
    
    def __init__(self, in_shape, out_shape, prm, device):
        
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
                                               
        self.learning_rate = prm['lr'] if 'lr' in prm else 0.01
        self.momentum = prm['momentum'] if 'momentum' in prm else 0.9
        self.batch_size = prm['batch'] if 'batch' in prm else 64
        self.transform = prm['transform'] if 'transform' in prm else None
        self.num_epochs = prm['epoch'] if 'epoch' in prm else 10
        
        self.criteria = nn.CrossEntropyLoss()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M'])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(6*6*128, 128),
                                        nn.ReLU(True),
                                        nn.Linear(128, out_shape[0]))
        
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
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
    
    def learn(self, train_data):
        self.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_data, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_data)))