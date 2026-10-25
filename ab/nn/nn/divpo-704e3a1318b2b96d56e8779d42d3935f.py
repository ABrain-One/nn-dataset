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
        # Fuse conv, bn, and relu for better efficiency
        assert self.training is False
        if self.conv.in_channels == self.conv.out_channels:
            return self.conv

        fused = nn.Sequential(
            nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            ),
            nn.BatchNorm2d(self.conv.out_channels)
        )

        w_conv = self.conv.weight.detach()
        w_bn = self.bn.weight.detach()
        b_bn = self.bn.bias.detach()
        eps = self.bn.eps

        w_conv_dim = list(w_conv.shape)
        w_bn_dim = list(w_bn.shape)

        # Match dimensions if they are not already matched
        if w_conv_dim[0] != w_bn_dim[0]:
            w_bn = w_bn.unsqueeze(-1).unsqueeze(-1).expand(w_conv_dim[0], -1, w_conv_dim[2], w_conv_dim[3])
        if w_conv_dim[1] != w_bn_dim[1]:
            w_conv = w_conv.unsqueeze(1).expand(-1, w_bn_dim[1], -1, -1, -1)

        fused[0].weight.data = (w_conv * (w_bn / (torch.sqrt(w_bn**2 + eps))))
        fused[0].bias.data = torch.mean(self.conv.bias.detach() + b_bn * (w_conv / (torch.sqrt(w_bn**2 + eps))), dim=1)

        fused[1].weight.data = w_bn
        fused[1].bias.data = b_bn

        return fused

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # adheres to the constraint
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = Block(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block3 = Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()