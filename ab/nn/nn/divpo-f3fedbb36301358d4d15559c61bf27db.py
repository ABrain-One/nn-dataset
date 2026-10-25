import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def fuse(self):
        if isinstance(self.conv[1], nn.BatchNorm2d):
            fused = nn.Sequential(
                nn.Conv2d(self.conv[0].in_channels,
                          self.conv[0].out_channels,
                          kernel_size=3,
                          stride=self.conv[0].stride,
                          padding=1),
                self.conv[1],
                nn.ReLU(inplace=True)
            )
            fused[0].weight.data = self.conv[0].weight.data
            fused[1].weight.data = self.conv[1].weight.data
            fused[1].bias.data = self.conv[1].bias.data + self.conv[1].running_mean * self.conv[1].weight.data
            return fused
        return self.conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive input channels
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block1 = Block(32, 64, stride=1)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

        self._set_training_props(prm)

    def _set_training_props(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self._set_training_props(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()