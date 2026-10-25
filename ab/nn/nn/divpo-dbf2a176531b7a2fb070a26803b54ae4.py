import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self._is_training = False
        self.to(device)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_optimizer(self, prm, param_groups):
        opt_class = torch.optim.SGD
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return opt_class(param_groups, lr=lr, momentum=momentum)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_setup(self, prm):
        self._is_training = True
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        for pg in self.optimizer.param_groups:
            pg['lr'] = prm.get('lr', 0.01)
        self.to(self.device)

    def learn(self, train_data):
        if not self._is_training:
            self.train_setup({})
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}