import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def on_epoch_end(self):
        self.stem.apply(_enable_bn)
        self.layer1.apply(_enable_bn)
        self.layer2.apply(_enable_bn)

    def _get_hyper_params(self, prm):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return lr, momentum

    def learn_from_train_data(self, train_data, prm):
        lr, momentum = self._get_hyper_params(prm)
        self.train_setup(prm)
        self.learn(train_data)
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[0]['momentum'] = momentum
        self.on_epoch_end()

    def learn_from_train_data_small(self, train_data, prm):
        lr, momentum = self._get_hyper_params(prm)
        self.train_setup(prm)
        self.learn(train_data)
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[0]['momentum'] = momentum
        self.on_epoch_end()

    def learn_from_train_data_large(self, train_data, prm):
        lr, momentum = self._get_hyper_params(prm)
        self.train_setup(prm)
        self.learn(train_data)
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[0]['momentum'] = momentum
        self.on_epoch_end()

def _enable_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.train()