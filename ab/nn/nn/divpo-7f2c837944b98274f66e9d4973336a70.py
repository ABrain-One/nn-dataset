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
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.build_layer(32, 32)
        self.layer2 = self.build_layer(32, 64)
        self.layer3 = self.build_layer(64, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(128 * 7 * 7, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = nn.functional.adaptive_avg_pool2d(x, 7)
        x = x.flatten(1)
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
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()

    def build_features(self):
        children = []
        return nn.Sequential(*children)

    def build_layer(self, in_c, out_c):
        return self._make_layer(in_c, out_c, blocks=2, stride=1)

    def _make_layer(self, in_c, out_c, blocks, stride):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        layers = []
        layers += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(out_c)]
        layers += [nn.ReLU(inplace=True)]
        in_c = out_c
        for i in range(1, blocks):
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)