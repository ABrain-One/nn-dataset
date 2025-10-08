import torch
from torch import nn

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_channels = 64
        self.dense_units = 64
        self.num_classes = out_shape[0] if len(out_shape) > 0 else 1
        self.learning_rate = prm['lr'] if 'lr' in prm else 0.01
        self.momentum = prm['momentum'] if 'momentum' in prm else 0.9
        self.dropout = prm['dropout'] if 'dropout' in prm else 0.5
        self.device = device
        
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.in_channels * 7 * 7, self.dense_units),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.dense_units, self.num_classes),
        )

        self.criteria = nn.CrossEntropyLoss()
    
    def _make_layers(self):
        layers = []
        layers += [nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1)]
        layers += [nn.BatchNorm2d(self.in_channels)]
        layers += [nn.ReLU(True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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