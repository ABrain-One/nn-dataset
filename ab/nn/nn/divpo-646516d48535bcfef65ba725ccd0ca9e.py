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
            nn.MaxPool2d(kernel_size=2),
            self._make_layer(32, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            self._make_classifier(256)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.avgpool(x)
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

    @staticmethod
    def _make_layer(width: int, blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(blocks):
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2))
            layers.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_classifier(width: int) -> nn.Sequential:
        layers = [nn.Dropout(0.5)]
        layers.append(nn.Linear(width, 10))
        return nn.Sequential(*layers)