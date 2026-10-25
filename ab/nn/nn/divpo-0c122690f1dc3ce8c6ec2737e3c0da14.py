import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (N, C, H, W)
        num_classes = int(out_shape[0])
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = self.make_layer(32, 64, 3, 1)
        self.layer2 = self.make_layer(64, 128, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))

    def make_layer(self, inplanes, outplanes, blocks, stride):
        layers = [
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()


# Single-branch variant (channels-first mode):
class Net_v1(Net):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net_v1, self).__init__(in_shape, out_shape, prm, device)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        # Channel-first input: (N, C, H, W)
        B = x.size(0)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
    supported_hyperparameters = supported_hyperparameters
    param_count = lambda: sum(p.numel() for p in self.parameters())