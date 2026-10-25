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
        # Fuse conv and bn for faster inference
        fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
            bias=True,
        ).requires_grad_(False)

        fused_conv.weight.data = (self.conv.weight.data / self.bn.weight.data.view(-1, 1, 1, 1)) * self.bn.running_var.view(-1, 1, 1, 1)
        fused_conv.weight.data *= self.bn.running_mean.view(-1, 1, 1, 1)
        fused_conv.bias.data = self.conv.bias.data - (self.bn.weight.data.view(-1) * self.bn.running_mean.view(-1)) / (self.bn.running_var.view(-1) + 1e-5)
        fused_conv.bias.data += self.bn.bias.data
        self.conv = fused_conv
        self.bn = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers = nn.Sequential(
            Block(32, 32),
            Block(32, 32),
            nn.Dropout(p=prm.get('dropout', 0.0)),
            Block(32, 64, stride=2),
            Block(64, 64),
            Block(64, 64),
            nn.Dropout(p=prm.get('dropout', 0.0)),
            Block(64, 128, stride=2),
            Block(128, 128),
            Block(128, 128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()