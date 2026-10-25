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
        self.layer1 = self.build_layer(32, 32, stride=1)
        self.layer2 = self.build_layer(32, 64, stride=2)
        self.layer3 = self.build_layer(64, 128, stride=2)
        self.layer4 = self.build_resblock(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._create_loss_func(prm)
        self.to(device)

    def build_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def build_resblock(self, in_channels, out_channels):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        ]
        if in_channels != out_channels:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            ] + layers
        layers += [nn.ReLU(inplace=True)]
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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _create_loss_func(self, prm):
        if not prm.get('loss_func', False):
            return
        loss_type = prm['loss_func']
        if loss_type.startswith('weighted'):
            class_weights = prm.get('class_weights', [1] * 10)
            class_weights = torch.tensor(class_weights).to(self.device)
            self.criteria = (nn.CrossEntropyLoss(weight=class_weights).to(self.device),)
        elif loss_type == 'bce':
            pos_weight = prm.get('pos_weight', 1.0)
            self.criteria = (nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device),)
        elif loss_type == 'focal':
            gamma = prm.get('gamma', 2.0)
            class_weights = prm.get('class_weights', [1] * 10)
            class_weights = torch.tensor(class_weights).to(self.device)
            self.criteria = (nn.CrossEntropyLoss(weight=class_weights).to(self.device),)

    def supported_hyperparameters(self):
        return ['lr', 'momentum', 'loss_func', 'class_weights', 'pos_weight', 'gamma']

def supported_hyperparameters():
    return ['lr', 'momentum', 'loss_func', 'class_weights', 'pos_weight', 'gamma']