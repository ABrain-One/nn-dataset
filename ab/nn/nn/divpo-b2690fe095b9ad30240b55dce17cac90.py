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
        # Fuse batchnorm and ReLU for better efficiency
        if self.training:
            return self
        self.conv = nn.utils.fuse_conv_bn_eval(self.conv, self.batchnorm)
        del self.batchnorm
        self.__dict__.update(self.__dict__.copy())
        self.__dict__.update({k: v for k, v in self.conv.__dict__.items() if k not in {'weight', 'bias', 'running_mean', 'running_var', 'eps', 'momentum', 'training'}})
        return self

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Handle channels-first input shape
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(in_channels, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(512, 512, stride=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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
            self.optimizer.step()

    def fuse(self):
        # Apply block-wise fusion to optimize for edge devices
        fused_blocks = []
        for blk in self.features:
            if isinstance(blk, Block) and not isinstance(blk, nn.Sequential):
                fused_blocks.append(blk.fuse())
            else:
                fused_blocks.append(blk)
        self.features = nn.Sequential(*fused_blocks)