import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = self._conv(3, 36, 1, Padding=1, activation="BnReLU")
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = self._conv(36, 72, 1, Padding=1, activation="BnReLU")
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = self._conv(72, 144, 3, Padding=1, activation="BnReLU")
        self.maxpool3 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._linear = nn.Linear(144 * 6 * 6, 10)

        self.dropout = prm.get('dropout', 0.0)
        if self.dropout > 0:
            self.dropout = nn.Dropout2d(p=self.dropout)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.avgpool(x)
        x = x.view(-1, 144 * 6 * 6)
        x = self._linear(x)
        return x

    def _conv(self, in_channels, out_channels, kernel_size, groups=1, stride=1, bias=True, Padding=0,
              activation=None):
        apad = int(Padding)
        padder = [apad, apad, apad, apad] if apad != 0 else []
        if len(padder):
            pad = nn.ZeroPad2d(padder)
        else:
            pad = nn.Identity()
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=groups,
                stride=stride,
                bias=bias,
                padding=0,
            ),
            pad,
            nn.BatchNorm2d(out_channels) if activation == "BnReLU" else nn.Identity(),
            nn.ReLU(inplace=True) if activation == "BnReLU" else nn.Identity(),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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