import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly latency
        if isinstance(self.conv, nn.Conv2d) and isinstance(self.batchnorm, nn.BatchNorm2d):
            fused_conv = nn.utils.fuse_conv_bn_eval(self.conv, self.batchnorm)
            self.__class__ = (
                Block if self.relu is not None else FusedBlock)  # Keep block name but replace class
            self.__dict__.update(self.__dict__)  # Copy properties
            self.conv = fused_conv  # Set to fused version
            del self.batchnorm, self.relu  # Remove original
            delattr(self, "batchnorm"), getattr(self, "relu", None)

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

class FusedBlock(Block):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        fused_conv = nn.utils.fuse_conv_bn_eval(self.conv, self.batchnorm)
        self.conv = fused_conv
        del self.batchnorm, self.relu  # Remove original
        delattr(self, "batchnorm"), getattr(self, "relu", None)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        self.num_classes = int(out_shape[0])
        self.stem = Block(self.in_channels, 64, kernel_size=3, padding=1)
        self.layer1 = Block(64, 64, stride=1)
        self.layer2 = Block(64, 128, stride=2, padding=0)
        self.layer3 = Block(128, 128, stride=1)
        self.layer4 = Block(128, 256, stride=2, padding=0)
        self.layer5 = Block(256, 256, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward_old(self, *args, **kwargs):
        return self.old_forward(*args, **kwargs)
    def train_setup_old(self, *args, **kwargs):
        return self.old_train_setup(*args, **kwargs)
    def learn_old(self, *args, **kwargs):
        return self.old_learn(*args, **kwargs)
    def old_forward(self, x):
        return self.forward(x)
    def old_train_setup(self, prm):
        self.train_setup(prm)
    def old_learn(self, train_data):
        self.learn(train_data)