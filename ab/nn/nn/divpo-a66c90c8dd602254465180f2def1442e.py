import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = self.in_channels
        self.features_layer = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.stage1 = self._make_layer(32, 32, 3, stride=1)
        self.stage2 = self._make_layer(32, 64, 3, stride=2)
        self.stage3 = self._make_layer(64, 128, 3, stride=2)
        self.stage4 = self._make_layer(128, 256, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.pre_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 6 * 6, 256)
        )
        self.classifier = nn.Linear(256, self.num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.features_layer.weight)
        nn.init.constant_(self.features_layer.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def features_forward(self, x):
        x = self.features_layer(x)
        x = self.stage1(x)
        return self._features_forward(x, 32, 32)

    def _features_forward(self, x, output_size, origin_size):
        return torch.nn.functional.adaptive_avg_pool2d(x, output_size=output_size)

    def forward(self, x):
        x = self.features_forward(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.pre_classifier(x)
        return self.classifier(x)

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