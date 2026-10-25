import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, stride=1)
        self.layer2 = self.build_layer(32, 64, 2, stride=2)
        self.layer3 = self.build_layer(64, 128, 2, stride=2)
        self.layer4 = self.build_resblock(128, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False
        self.to(device)

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True)]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def build_resblock(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        if self._is_fwd_comp:
            return
        self._is_fwd_comp = True
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
        self._is_fwd_comp = False

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def cross_entropy_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    def accuracy(self, outputs, labels):
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)

    def learn_from_train_data(self, train_data, prm):
        self.train()
        self.train_setup(prm)
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.learn((inputs, labels))
        self.optimizer = None
        self.criteria = None
        self.to_empty()