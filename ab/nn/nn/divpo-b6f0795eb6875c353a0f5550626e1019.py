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
        self.features = 32
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1, bias=False)
        self.bn_first_conv = nn.BatchNorm2d(self.features)
        self.maxpool = nn.MaxPool2d(2)
        self.encoder1 = self._encoder_block(32, 32, identity_downsample=True)
        self.encoder2 = self._encoder_block(32, 64, identity_downsample=True)
        self.encoder3 = self._encoder_block(64, 128, identity_downsample=True)
        self.encoder4 = self._encoder_block(128, 256, identity_downsample=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)

    def _encoder_block(self, input_channels, output_channels, identity_downsample=None):
        expansion = 1
        layers = []
        layers += [
            nn.Conv2d(input_channels, output_channels * expansion, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels * expansion),
            nn.ReLU(inplace=True)
        ]
        self.features = output_channels * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn_first_conv(x)
        x = self.maxpool(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()