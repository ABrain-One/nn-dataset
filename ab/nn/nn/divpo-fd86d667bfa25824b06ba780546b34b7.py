import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0)
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim)
        self.pointwise_conv = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.use_res_connect:
            out = x
        else:
            out = torch.nn.functional.pad(x, [0, 1, 0, 1], mode='constant', value=0)

        out = self.expand_conv(out)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)

        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = 32
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 1e-3)
        self.momentum = prm.get('momentum', 0.9)

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.inverted1 = InvertedResidual(32, 32, 2, 6)
        self.inverted2 = InvertedResidual(32, 64, 2, 6)
        self.inverted3 = InvertedResidual(64, 128, 2, 6)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inverted1(x)
        x = self.inverted2(x)
        x = self.inverted3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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