import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deploy=False, act_layer=nn.ReLU):
        super().__init__()
        self.deploy = deploy
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.act = act_layer(inplace=True)

        self.conv_rep = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels, bias=False)
        self.bn_rep = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        if not self.deploy:
            x = self.act(self.bn(self.conv(x)))
            x = self.conv_rep(x)
            x = self.bn_rep(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            x = self.conv_rep(x)
            x = self.bn_rep(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        layers = []
        channels = [32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(
                RepVGGBlock(
                    channels[i], channels[i + 1],
                    kernel_size=3, stride=2, padding=1,
                    deploy=False, act_layer=nn.ReLU
                )
            )
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()