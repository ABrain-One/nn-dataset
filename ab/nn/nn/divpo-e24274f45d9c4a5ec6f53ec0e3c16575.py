import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

    def _make_layer(self, inplanes, planes, blocks):
        layers = [
            nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ]
        for _ in range(blocks):
            layers += [
                nn.Sequential(
                    nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True)
                )
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 1e-4)
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


# The following code is an example of how you might generate a dummy dataset for testing purposes.
# You should replace this with actual data loading code for your dataset.
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=32, length=1000):
        self.len = length
        self.img_size = (3, size, size)
        self.data = torch.randn(length, *self.img_size)
        self.labels = torch.randint(0, 10, (length,), dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len