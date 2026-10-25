import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly deployment
        assert self.training is False
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        self.batchnorm.momentum = None
        fused = torch.nn.utils.fuse_conv_batch_norm_eval(self.conv, self.batchnorm)
        fused = Block(self.conv.in_channels, self.conv.out_channels,
                      self.conv.kernel_size, self.conv.stride, self.conv.padding)
        fused.conv = torch.nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                                     self.conv.kernel_size, self.conv.stride, self.conv.padding,
                                     self.conv.dilation, self.conv.groups,
                                     False, self.conv.weight.dtype)
        fused.conv.weight.copy_(self.conv.weight)
        fused.batchnorm.copy_from(self.batchnorm)
        self.__class__ = torch.nn.Conv2d
        self.__name__ = 'fused_' + self.__name__
        return fused

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape as (batch, channels, height, width)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.layer4 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()