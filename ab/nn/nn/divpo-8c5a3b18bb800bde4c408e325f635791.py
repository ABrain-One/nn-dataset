import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, affine):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # t channel
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=in_channels,
                bias=False,
                dilation=1,
                padding_mode='zeros',
                device=None,
                dtype=None
            ),
            nn.BatchNorm2d(in_channels * expand_ratio, affine=affine),
            nn.ReLU6(inplace=True),
            # t+3x3+t conv
            nn.Conv2d(
                in_channels=in_channels * expand_ratio,
                out_channels=in_channels * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expand_ratio,
                bias=False,
                dilation=1,
                padding_mode='zeros',
                device=None,
                dtype=None
            ),
            nn.BatchNorm2d(in_channels * expand_ratio, affine=affine),
            nn.ReLU6(inplace=True),
            # t channel
            nn.Conv2d(
                in_channels=in_channels * expand_ratio,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                dilation=1,
                padding_mode='zeros',
                device=None,
                dtype=None
            ),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1, False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 64, 1, 6, False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(64, 128, 1, 6, False),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
            outputs = self.forward(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()