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
        self.features = 32
        self.body = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.layer1 = self.build_layer(32, 32, 1)
        self.layer2 = self.build_layer(32, 64, 2)
        self.layer3 = self.build_layer(64, 128, 3)
        self.layer4 = self.build_layer(128, 256, 4)
        self.gap = nn.AdaptiveAvgPool2d((6, 6))
        self.cls = nn.Linear(256 * 6 * 6, self.num_classes)

    def build_layer(self, in_filter, out_filter, block_id):
        layers = []
        ds_strides = 1
        ds_filters = in_filter
        if block_id:
            ds_strides = 2
            ds_filters = out_filter
        layers += [
            nn.Conv2d(self.features, ds_filters, kernel_size=ds_strides, stride=ds_strides),
            nn.BatchNorm2d(ds_filters)]
        self.features = out_filter
        for idx in range(0, block_id):
            layers += [
                nn.Conv2d(self.features, self.features, kernel_size=1),
                nn.BatchNorm2d(self.features),
                nn.ReLU(True),
                nn.Conv2d(self.features, self.features, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.features),
                nn.ReLU(True),
                nn.Conv2d(self.features, self.features, kernel_size=1),
                nn.BatchNorm2d(self.features),
                nn.ReLU(True)]
            self.features = out_filter
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = nn.functional.adaptive_avg_pool2d(x, 6)
        x = x.flatten(1)
        x = self.cls(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()