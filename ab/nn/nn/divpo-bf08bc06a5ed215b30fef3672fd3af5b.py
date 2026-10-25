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
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_federated = False
        self._num_federated_clients = 0
        self._num_federated_epochs = 0
        self._num_federated_samples = 0
        self._num_federated_updates = 0
        self._num_federated_parameters = 0
        self._num_federated_commits = 0
        self._num_federated_restarts = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
        layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(num_blocks - 1):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
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
        self.to(self.device)
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
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return ['lr', 'momentum']

    def auto_adjust_learning_rate(self, epoch):
        if epoch < 10:
            lr = 0.01
        elif epoch < 20:
            lr = 0.001
        else:
            lr = 0.0001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self._is_federated = prm.get('_is_federated', False)
        self._num_federated_clients = prm.get('_num_federated_clients', 0)
        self._num_federated_epochs = prm.get('_num_federated_epochs', 0)
        self._num_federated_samples = prm.get('_num_federated_samples', 0)
        self._num_federated_updates = prm.get('_num_federated_updates', 0)
        self._num_federated_parameters = prm.get('_num_federated_parameters', 0)
        self._num_federated_commits = prm.get('_num_federated_commits', 0)
        self._num_federated_restarts = prm.get('_num_federated_restarts', 0)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def update_federated_stats(self, num_clients, num_epochs, num_samples, num_updates, num_parameters, num_commits, num_restarts):
        self._num_federated_clients = num_clients
        self._num_federated_epochs = num_epochs
        self._num_federated_samples = num_samples
        self._num_federated_updates = num_updates
        self._num_federated_parameters = num_parameters
        self._num_federated_commits = num_commits
        self._num_federated_restarts = num_restarts