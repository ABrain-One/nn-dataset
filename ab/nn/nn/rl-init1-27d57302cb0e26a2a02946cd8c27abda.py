import torch
from torch import nn

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']

class Net(nn.Module):
    
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device

        self.learning_rate = self.prm['lr'] if 'lr' in self.prm else 0.001
        self.momentum = self.prm['momentum'] if 'momentum' in self.prm else 0.9
        self.dropout = self.prm['dropout'] if 'dropout' in self.prm else 0.5
        self.batch_size = self.prm['batch'] if 'batch' in self.prm else 32

        self.features = self._make_layers([32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M'])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, self.out_shape[0])
        )

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
        layers += [nn.AdaptiveAvgPool2d((6, 6))]
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
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            if (i+1) % self.batch_size == 0:   
                self.scheduler.step()