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
        # Fuse Conv2d + BN + ReLU into one Conv2d
        fused_conv = (self.conv.weight / self.bn.running_var.reshape(-1, 1, 1, 1)).sqrt() * self.conv.weight * (self.bn.weight.reshape(-1, 1, 1, 1) / self.bn.running_var.reshape(-1, 1, 1, 1).sqrt())
        bias = self.bn.bias - self.bn.weight.reshape(-1) * self.bn.running_mean / self.bn.running_var.reshape(-1) + self.conv.bias if self.conv.bias is not None else self.bn.bias - self.bn.weight.reshape(-1) * self.bn.running_mean / self.bn.running_var.reshape(-1)
        self.fused_conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, bias=True)
        self.fused_conv.weight.data = fused_conv
        self.fused_conv.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.__delattr__('relu')
        self.conv = self.fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) - derive input channels from in_shape
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 64, stride=1)
        self.block4 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self._set_training_mode(prm.get('mode', 'train'))

    def _set_training_mode(self, mode):
        self.train()
        if mode == 'train':
            self.block1.train()
            self.block2.train()
            self.block3.train()
            self.block4.train()
            self.avgpool.train()
        elif mode == 'fuse':
            self.block1.eval()
            self.block2.eval()
            self.block3.eval()
            self.block4.eval()
            self.avgpool.eval()
            self.block1.fuse()
            self.block2.fuse()
            self.block3.fuse()
            self.block4.fuse()

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self._set_training_mode(prm.get('mode', 'train'))
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        params = list(self.classifier.parameters())
        self.optimizer = torch.optim.SGD(
            params,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()