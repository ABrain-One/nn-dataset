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
        # Fused block for edge-friendly deployment
        assert self.training is False
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
            bias=True
        ).requires_grad_(False)
        fused_conv.weight.data = self.bn.weight.data.view(1, -1, 1, 1) / (self.bn.running_var.data + self.bn.eps).sqrt()
        fused_conv.weight.data *= self.bn.running_mean.data.view(1, -1, 1, 1)
        fused_conv.bias.data = self.bn.bias.data - self.bn.weight.data.view(-1) * self.bn.running_mean.data / (self.bn.running_var.data + self.bn.eps).sqrt()
        fused_bn = nn.BatchNorm2d(self.conv.out_channels, affine=False).requires_grad_(False)
        fused_bn.weight.data = torch.ones(self.conv.out_channels)
        fused_bn.bias.data = torch.zeros(self.conv.out_channels)
        self._modules['conv'] = fused_conv
        self._modules['bn'] = fused_bn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape for CIFAR-10 (should be 3 for RGB)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 64, stride=1)
        self.block4 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

# Auto-generated template for CIFAR-10 model with max parameters <= 500000
in_shape = (3, 32, 32)  # channels-first (C, H, W) format
out_shape = (10,)
prm = {'lr': 0.01, 'momentum': 0.9}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(in_shape, out_shape, prm, device)
model.train()