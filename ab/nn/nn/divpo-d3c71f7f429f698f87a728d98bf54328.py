import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self._make_layer(32, 64, 2, True)
        self.layer2 = self._make_layer(64, 128, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        lr, momentum = map(prm.get, ['lr', 'momentum'])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=0.0001)

    def _make_layer(self, inplanes, planes, blocks, downsample):
        layers = [nn.Conv2d(inplanes, planes, kernel_size=3, stride=1 if not downsample else 2, padding=1)]
        layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        if downsample: layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()


# Export the named class to define the module.
class Net(Net):
    def get_flops(self, input_shape):
        # Prepare input tensor
        x = torch.empty(input_shape)
        # Count FLOPs
        flops = torch.profile(self, inputs=(x,), custom_ops={}, verbose=False).total_ops
        # Return FLOPs as integer
        return int(flops)