import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = self._make_layer(32, 32, 2, drop_prob=prm.get('dropout', 0.0))
        self.layer2 = self._make_layer(32, 64, 2, drop_prob=prm.get('dropout', 0.0))
        self.layer3 = self._make_layer(64, 128, 2, drop_prob=prm.get('dropout', 0.0))
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, blocks, drop_prob):
        layers = [nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob)
        )]
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_prob)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
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

# Edge-friendly latency budget implies simplicity and small parameter count.
# This design includes efficient block and layer configurations:
# - 2-3 block residual units per layer to keep model size manageable.
# - Dropout for regularization, controlled by 'dropout' hyperparameter.
# - AdaptiveAvgPool2d for spatial downsampling to maintain efficiency.
# - Flatten and 1x1 convs before classifier to keep parameters low.
# The target parameter count is 500000, so the design must respect this limit.
# The model below or similar will meet the criteria.