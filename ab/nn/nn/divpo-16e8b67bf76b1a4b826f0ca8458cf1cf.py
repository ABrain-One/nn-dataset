import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused weights and batch norm for better efficiency
        fused_conv = (self.conv.weight / self.bn.running_var.sqrt()).view(*self.conv.weight.shape)
        fused_bias = self.conv.bias - self.bn.running_mean * self.bn.weight / self.bn.running_var.sqrt() + self.bn.bias
        self.fused_conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                                    kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                                    padding=self.conv.padding, bias=True)
        self.fused_conv.weight.data = fused_conv
        self.fused_conv.bias.data = fused_bias
        self.__delattr__('bn')
        self.__delattr__('conv')
        self.conv = self.fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=2),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=2),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
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