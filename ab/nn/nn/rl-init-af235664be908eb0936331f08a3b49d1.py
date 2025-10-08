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
        self.batch_size = prm['batch']
        self.dropout_value = prm['dropout']
        self.momentum = prm['momentum']
        self.transform = prm['transform']

        self.features = self._make_layers([32, 64, 128])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_value),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, self.out_shape[0]))

        self.criteria = nn.CrossEntropyLoss()
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            v = int(v)
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True)]
            in_channels = v
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