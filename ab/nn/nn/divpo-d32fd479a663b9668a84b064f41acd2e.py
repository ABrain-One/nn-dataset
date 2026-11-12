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
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self._make_layer(32, 32, 1),
            self._make_layer(32, 64, 2),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(64 * 8 * 8, self.num_classes)

    def _make_layer(self, in_planes, planes, num_blocks):
        layers = []
        layers.append(self._encode_basic_block(in_planes, planes))
        for _ in range(1, num_blocks):
            layers.append(self._encode_basic_block(planes, planes))
        return nn.Sequential(*layers)

    def _encode_basic_block(self, in_planes, planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.AdaptiveAvgPool2d((8, 8))(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()