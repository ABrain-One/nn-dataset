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
        self.features = 32
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.features = 32
        self.layer1 = self.build_layer(32, 32)
        self.layer2 = self.build_layer(32, 64)
        self.layer3 = self.build_layer(64, 128)
        self.layer4 = self.build_layer(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def build_layer(self, in_filter, out_filter, is_first=False):
        x = nn.Sequential()
        pading = 1
        kernel = 3
        x.add_module('conv1', nn.Conv2d(in_filter, out_filter, kernel_size=kernel, padding=pading, bias=False))
        x.add_module('bat1', nn.BatchNorm2d(out_filter))
        x.add_module('act1', nn.ReLU(True))
        x.add_module('conv2', nn.Conv2d(out_filter, out_filter, kernel_size=kernel, padding=pading, bias=False))
        x.add_module('bat2', nn.BatchNorm2d(out_filter))
        self.shortcut = nn.Sequential()
        if is_first==False:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_filter, out_filter, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_filter)
            )
        else:
            self.conv1=self.conv2=nn.Identity()
        self.act=nn.ReLU(True)
        self.out_filter = out_filter
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
            self.optimizer.step()