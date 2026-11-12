import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.prms = prm
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_layer(32, 32, 1, 0.3)
        self.block2 = self._make_layer(32, 64, 1, 0.3)
        self.classifier = nn.Linear(64, self.num_classes)
        self.criteria = nn.CrossEntropyLoss().to(self.device)

    def _make_layer(self, in_channels, out_channels, blocks, drop_out):
        layers = []
        self.in_channels = in_channels
        layers.append(self._conv_layer(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=drop_out))
        for i in range(1, blocks):
            layers.append(self._conv_layer(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _conv_layer(self, in_c, out_c, kernel_size, padding):
        return nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.relu(x.mean(dim=(2, 3)))
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = self.criteria.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()