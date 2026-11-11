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
        self.features = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.v1 = self._build_v(32, 32, 32)
        self.v2 = self._build_v(32, 32, 32)
        self.v3 = self._build_v(32, 8, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, self.num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                nn.init.constant_(m.bias.data, 0)

    def _build_v(self, input_size, kernel_size, output_size):
        layers = []
        layers += [
            nn.Conv2d(input_size, kernel_size, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(kernel_size),
            nn.ReLU(inplace=False),
            nn.Conv2d(kernel_size, kernel_size, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size, output_size, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        ]
        self.layers = nn.Sequential(*layers)
        return self.layers

    def forward(self, x):
        x = self.features(x)
        x1 = self.v1(x)
        x2 = self.v2(x1)
        x3 = self.v3(x2)
        avg_x3 = self.avgpool(x3)
        avg_x3 = avg_x3.squeeze(3).squeeze(2)
        classifer = self.classifier(avg_x3)
        return classifer

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