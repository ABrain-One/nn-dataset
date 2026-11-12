import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = self._make_layer(32, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = self._make_layer(64, 128, stride=2, kernel_size=3, padding=1)
        self.layer4 = self._make_layer(128, 256, stride=2, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def build_features(self):
        return nn.Sequential()

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        return nn.utils.spectral_norm(layer)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum, 
            nesterov=True, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.scheduler.step()