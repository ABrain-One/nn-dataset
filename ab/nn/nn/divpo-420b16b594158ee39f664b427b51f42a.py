import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        if not (self.conv.weight.shape[2:] == self.bn.running_mean.shape):
            raise ValueError("Fusing requires input and BN to be the same shape.")
        w = self.conv.weight.clone()
        mean = self.bn.running_mean
        var_sqrt = torch.sqrt(self.bn.running_var + self.bn.eps)
        w *= (mean / var_sqrt).reshape((-1, 1, 1, 1))
        b = -mean * w.sum(dim=(2, 3), keepdim=True) + self.bn.bias.reshape(-1, 1, 1, 1)
        self.conv = torch.nn.Conv2d(in_channels=w.shape[1], out_channels=w.shape[0], kernel_size=w.shape[2], stride=self.conv.stride, padding=self.conv.padding, groups=w.shape[0])
        self.conv.weight.data = w
        self.conv.bias.data = torch.flatten(b)
        delattr(self, "bn")
        self.__delattr__("bn")

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape for RGB input
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(64, 64, stride=1)
        self.layer2 = Block(64, 128, stride=2)
        self.layer3 = Block(128, 128, stride=1)
        self.layer4 = Block(128, 256, stride=2)
        self.layer5 = Block(256, 256, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
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