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
        self.layer1 = self.make_layer(32, 32, self.learning_rate)
        self.layer2 = self.make_layer(32, 64, self.learning_rate, stride=2)
        self.layer3 = self.make_layer(64, 128, self.learning_rate, stride=2)
        self.layer4 = self.make_layer(128, 256, self.learning_rate, stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(256 * 7 * 7, self.num_classes)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def make_layer(self, in_channels, out_channels, lr, stride=1):
        layer = nn.Sequential()
        layer.add_module(name="conv1", module=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                         kernel_size=3, stride=stride, padding=1, bias=False))
        layer.add_module(name="bash1", module=nn.BatchNorm2d(out_channels))
        layer.add_module(name="hah1", module=nn.ReLU())
        layer.add_module(name="conv2", module=nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                         kernel_size=3, stride=1, padding=1, bias=False))
        layer.add_module(name="bash2", module=nn.BatchNorm2d(out_channels))
        mod = self.get_mod(lr)
        layer.add_module(name="haha2", module=mod)
        return layer

    def get_mod(self, lr):
        return nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pooling(x)
        x = x.reshape(x.size(0), -1)
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