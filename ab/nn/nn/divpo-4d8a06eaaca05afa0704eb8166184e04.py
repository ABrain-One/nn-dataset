import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        if hasattr(self.bn, 'weight') and self.bn.weight is not None:
            bn_w = self.bn.weight.detach()
            bn_b = self.bn.bias.detach() if self.bn.bias is not None else torch.zeros_like(bn_w)
            w = self.conv.weight.detach()

            conv_w = torch.nn.functional.conv2d(
                w * (bn_w.view(1, -1, 1, 1)),
                w.new_zeros(1, 1, 3, 3).fill_(1),
                groups=w.shape[0],
            )
            conv_b = torch.nn.functional.conv2d(
                w * (bn_b.view(1, -1, 1, 1)),
                w.new_zeros(1, 1, 3, 3).fill_(1),
                groups=w.shape[0],
            )

            self.conv.weight.data.copy_(conv_w)
            if self.conv.bias is not None:
                self.conv.bias.data.copy_(conv_b.view(-1))
            if self.bn.bias is not None:
                self.bn.weight.data.copy_(bn_w)
                self.bn.bias.data.copy_(bn_b)
            self.__class__ = nn.Conv2d
            self.bn = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Handle channels-first input (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()