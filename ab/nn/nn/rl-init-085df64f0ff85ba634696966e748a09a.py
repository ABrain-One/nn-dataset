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
        
        self.learning_rate = prm['lr'] if 'lr' in prm else 0.01
        self.momentum = prm['momentum'] if 'momentum' in prm else 0.9
        self.dropout = prm['dropout'] if 'dropout' in prm else 0.5
        self.batch_size = prm['batch'] if 'batch' in prm else 64

        self.features = self._make_layers([
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, self.out_shape[0]))

        self.criteria = nn.CrossEntropyLoss()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if isinstance(x, nn.Conv2d):
                layers += [x]
                in_channels = x.out_channels
            elif isinstance(x, nn.MaxPool2d):
                layers += [x]
            elif isinstance(x, nn.Dropout2d):
                layers += [x]
            elif isinstance(x, nn.ReLU):
                layers += [x]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

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