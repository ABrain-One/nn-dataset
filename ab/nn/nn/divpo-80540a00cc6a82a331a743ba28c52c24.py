import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, input_channel, output_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channel == output_channel

        hidden_channel = input_channel * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU6(inplace=True)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=hidden_channel, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU6(inplace=True)
        )
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_channel, output_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channel)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.project_conv(self.depthwise_conv(self.expand_conv(x)))
        else:
            return self.project_conv(self.depthwise_conv(self.expand_conv(x)))

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, **kwargs):
        super(Block, self).__init__()
        self.inverted_residual = InvertedResidual(in_channels, out_channels, stride, expand_ratio)

    def forward(self, x):
        return self.inverted_residual(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=32, out_channels=32, stride=1, expand_ratio=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=32, out_channels=64, stride=1, expand_ratio=6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(in_channels=64, out_channels=96, stride=1, expand_ratio=6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Flatten(),
            nn.Linear(96 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(512, out_shape[0])
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria[0](output, target)
            loss.backward()
            self.optimizer.step()