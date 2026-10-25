import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.out_channels = [32, 64, 128]
        self.out_channels_list = [self.out_channels[0]]
        self.num_classes = int(out_shape[0])
        self.layers = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels[0], stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn(self.out_channels[0], self.out_channels[1], stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn(self.out_channels[1], self.out_channels[2], stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.out_channels[2], self.num_classes),
        )

    def forward(self, x):
        return self.layers(x)

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