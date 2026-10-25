import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self.make_layer(32, 64, 2, 1)
        self.layer2 = self.make_layer(64, 128, 2, 2)
        self.layer3 = self.make_layer(128, 256, 2, 2)
        self.layer4 = self.make_layer(256, 512, 2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, out_shape[0])

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(blocks):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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


# Expected input size: 3 x 32 x 32
# Output size: 10
# Expected weights size: 32 x 3 x 3 x 3
# Expected weights size: 32
# Expected weights size: 64 x 32 x 3 x 3
# Expected weights size: 64
# Expected weights size: 64 x 64 x 3 x 3
# Expected weights size: 64
# Expected weights size: 128 x 64 x 3 x 3
# Expected weights size: 128
# Expected weights size: 128 x 128 x 3 x 3
# Expected weights size: 128
# Expected weights size: 256 x 128 x 3 x 3
# Expected weights size: 256
# Expected weights size: 256 x 256 x 3 x 3
# Expected weights size: 256
# Expected weights size: 512 x 256 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Expected weights size: 512 x 512 x 3 x 3
# Expected weights size: 512
# Total params: 502500 (200000 Trainable)