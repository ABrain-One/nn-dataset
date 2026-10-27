import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused version of Block, meant for latency-critical environments
        assert hasattr(self, 'batchnorm')
        self.conv = nn.utils.fuse_conv_and_bn(self.conv, self.batchnorm)
        delattr(self, 'batchnorm')

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2 = Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.block3 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self._set_training_mode(prm.get('lr', 0.01), prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.block2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _set_training_mode(self, lr, momentum):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum
        )

    def train_setup(self, prm):
        self._set_training_mode(prm.get('lr', 0.01), prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def fuse(self):
        # Apply fusion for improved latency
        for block in [self.block1, self.block2, self.block3]:
            if isinstance(block, Block):
                block.fuse()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))