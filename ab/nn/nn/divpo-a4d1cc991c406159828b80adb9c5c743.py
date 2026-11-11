import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, 3, padding=1)
        self.conv2 = self._make_layer(32, 32, 1)
        self.conv3 = self._make_layer(32, 64, 3, stride=2)
        self.conv4 = self._make_layer(64, 64, 3)
        self.conv5 = self._make_layer(64, 128, 3, stride=2)
        self.conv6 = self._make_layer(128, 128, 3)
        self.avgpool = self._global_avgpool()
        self.classifier = nn.Linear(128, self.num_classes)
        self.to(self.device)

    def _make_layer(self, in_filters: int, filters: int, blocks: int, stride: int = 1) -> nn.Sequential:
        block = []
        if stride != 1 or in_filters != filters:
            block += [nn.Conv2d(in_filters, filters, 1, stride=stride, bias=False)]
            block += [nn.BatchNorm2d(filters)]
        for i in range(blocks):
            block += [nn.Conv2d(filters, filters, 3, padding=1, bias=False)]
            block += [nn.BatchNorm2d(filters)]
            if i != blocks - 1:
                block += [nn.ReLU(inplace=True)]
        return nn.Sequential(*block)

    def _global_avgpool(self):
        return nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()