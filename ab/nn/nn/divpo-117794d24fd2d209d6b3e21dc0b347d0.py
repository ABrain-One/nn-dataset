import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self.make_layer(32, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

    def make_layer(self, in_channels, out_channels, blocks):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ]
        for _ in range(blocks - 1):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
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
            weight_decay=prm.get('weight_decay', 1e-4)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()


def check_shape_consistency(in_shape, out_shape, prm):
    # Check if in_shape is consistent with input channels
    expected_channels = 3 if len(in_shape) == 4 else in_shape[0]
    assert in_shape[1] == expected_channels, "Input channels mismatch"

    # Check if out_shape represents a valid number of classes
    assert out_shape[0].is_integer() and out_shape[0] > 0, "Invalid number of output classes"

    # Check if prm contains valid learning rate and momentum
    assert 'lr' in prm, "Missing learning rate parameter"
    assert 'momentum' in prm, "Missing momentum parameter"

    # Additional consistency checks (optional)
    assert isinstance(prm['lr'], (int, float)), "Learning rate should be a number"
    assert isinstance(prm['momentum'], (int, float)), "Momentum should be a number"
    assert 0 < prm['lr'] <= 1, "Learning rate should be within (0, 1]"
    assert 0 < prm['momentum'] < 1, "Momentum should be within (0, 1)"