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
        # Fused block for edge-friendly deployment
        if isinstance(self.bn, nn.BatchNorm2d):
            fused_conv = nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                bias=True
            )

            # Merge batch norm into conv layer weights
            bn_weight = self.bn.weight
            bn_bias = self.bn.bias
            bn_running_mean = self.bn.running_mean
            bn_running_var = self.bn.running_var
            if self.bn.affine:
                bn_weight_fused = bn_weight * (bn_running_var + bn_bias)
                bn_bias_fused = bn_bias * bn_running_mean
            else:
                bn_weight_fused = bn_weight
                bn_bias_fused = bn_bias
            fused_conv.weight.data = (bn_weight_fused.view(self.conv.out_channels, 1, 1, 1) * self.conv.weight.data).clone()
            fused_conv.bias.data = bn_bias_fused.clone()

            # Replace block with fused conv
            self.__class__ = nn.Module
            self.__dict__.update(fused_conv.__dict__)
            del self.conv
            del self.bn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # Handle channels-first input (C, H, W)
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(6)
        )

        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

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
            lr=prm.get('lr', self.learning_rate),
            momentum=prm.get('momentum', self.momentum)
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

    def get_parameters(self):
        return self.parameters()

    def fuse(self):
        for block in self.features:
            if isinstance(block, Block):
                block.fuse()
        self.features.rearrange()
        self.classifier.rearrange()