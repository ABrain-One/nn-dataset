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
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        allow_2nd_layer = True
        layers += [
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ][:allow_2nd_layer]

        layers += [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]

        self._first_block = nn.Sequential(*layers)

        # We won't instantiate downsample - we don't need it!
        # self.downsample1 = nn.Sequential(...)

        layers = []
        self._middle_block = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self._middle_block.weight, mode="fan_out", nonlinearity="relu")
        self._mid_batchnorm = nn.BatchNorm2d(128)
        self._mid_relu = nn.ReLU(inplace=True)
        # We won't instantiate downsample - we don't need it!
        # self.downsample2 = nn.Sequential(...)

        layers = []
        layers += [
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ][:allow_2nd_layer]

        self._last_block = nn.Sequential(*layers)

        self._global_avg_pool = None
        self._last_channels = 128

        # No dropout as it was not in original and our is small
        # self.drop = nn.Dropout(p=prm.get('dropout', 0.5))

    def forward(self, x):
        # Implement forward pass
        x = self._first_block(x)
        x = self._middle_block(x)
        x = self._last_block(x)

        base = self.avgpool(x)
        features = torch.flatten(base, 1)
        # Features are flattened but dropout is not applied
        # features = self.drop(features)
        log_probs = self.classifier(features)
        return log_probs

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