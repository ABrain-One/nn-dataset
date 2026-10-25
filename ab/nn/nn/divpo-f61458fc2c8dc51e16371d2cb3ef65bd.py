import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def fuse(self):
        if isinstance(self.conv[1], nn.BatchNorm2d):
            fused_conv = nn.Conv2d(
                self.conv[0].in_channels,
                self.conv[0].out_channels,
                kernel_size=self.conv[0].kernel_size,
                stride=self.conv[0].stride,
                padding=self.conv[0].padding,
                bias=True
            )

            # fused_conv.weight = self.conv[0].weight
            # fused_conv.bias = self.conv[0].bias
            # fused_conv.running_mean = self.conv[1].running_mean
            # fused_conv.running_var = self.conv[1].running_var
            # fused_conv.num_batches_tracked = self.conv[1].num_batches_tracked

            fused_conv.weight = nn.Parameter(self.conv[0].weight)
            fused_conv.bias = nn.Parameter(self.conv[0].bias)
            fused_conv.running_mean = nn.Parameter(self.conv[1].running_mean)
            fused_conv.running_var = nn.Parameter(self.conv[1].running_var)
            fused_conv.num_batches_tracked = nn.Parameter(self.conv[1].num_batches_tracked)

            fused_bn = nn.BatchNorm2d(
                fused_conv.out_channels,
                eps=self.conv[1].eps,
                momentum=self.conv[1].momentum
            )

            self.conv = nn.Sequential(
                fused_conv,
                fused_bn,
                nn.ReLU(inplace=True)
            )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.dropout = prm.get('dropout', 0.3)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256, stride=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
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

    def fuse(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.fuse()