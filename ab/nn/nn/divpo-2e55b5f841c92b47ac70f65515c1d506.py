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
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = 32
        self.body = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1, bias=False)
        self.layer1 = self.build_layer(32, 32, 1)
        self.layer2 = self.build_layer(32, 64, 2)
        self.layer3 = self.build_layer(64, 128, 2)
        self.layer4 = self.build_layer(128, 256, 2)
        self.gap = nn.AdaptiveAvgPool2d((6, 6))
        self.cls = nn.Linear(256 * 6 * 6, self.num_classes)
        self.initialize_weights()

    def build_layer(self, in_c: int, out_c: int, blocks: int, stride: int = 1) -> nn.Module:
        layers = []
        layers += [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        self.features = out_c
        for i in range(1, blocks):
            layers += [
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        new_x = self.body(x)
        new_x = self.layer1(new_x)
        new_x = self.layer2(new_x)
        new_x = self.layer3(new_x)
        new_x = self.layer4(new_x)
        new_x = self.gap(new_x)
        new_x = new_x.view(new_x.size(0), -1)
        new_x = self.cls(new_x)
        return new_x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()