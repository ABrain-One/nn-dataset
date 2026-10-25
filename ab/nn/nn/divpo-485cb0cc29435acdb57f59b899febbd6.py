import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.rbr_reparam = self.conv

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def extra_repr(self):
        return f'in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, ' \
               f'kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, ' \
               f'padding={self.conv.padding}, dilation={self.conv.dilation}, groups={self.conv.groups}'

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            RepVGGBlock(32, 32, stride=2, padding=1),
            RepVGGBlock(32, 32, stride=2, padding=1),
            RepVGGBlock(32, 32, stride=2, padding=1),
            RepVGGBlock(32, 32, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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