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
        self.create_layers()

    def create_layers(self):
        layers = []
        in_channels = self.in_channels

        # Initial convolutional layer
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # First block
        layers.extend(self._make_block(32, 16, 2))
        
        # Second block
        layers.extend(self._make_block(16, 32, 2))

        # Third block
        layers.extend(self._make_block(32, 64, 2))

        self.features = nn.Sequential(*layers)

        # Final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, self.num_classes)

    def _make_block(self, in_channels, out_channels, stride):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(in_channels))

        for _ in range(2):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()