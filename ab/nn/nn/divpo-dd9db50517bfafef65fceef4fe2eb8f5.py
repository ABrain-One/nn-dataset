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
        if hasattr(self.conv, 'weight'):
            conv_w = self.conv.weight.clamp_
            conv_b = self.conv.bias
            fused_w = conv_w
            if hasattr(self.batchnorm, 'weight'):
                bn_w = self.batchnorm.weight
                bn_b = self.batchnorm.bias
                eps = self.batchnorm.eps
                invstd = bn_w.clone().inverse().add_(eps).sqrt_()
                fused_w = conv_w * (bn_w.reshape(-1, 1, 1, 1) * invstd.reshape(1, -1, 1, 1)).reshape(conv_w.shape)
                fused_b = conv_b * bn_w * invstd + bn_b
            self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                                  kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                                  padding=self.conv.padding, bias=True)
            self.conv.weight.data.copy_(fused_w)
            self.conv.bias.data.copy_(fused_b)
            self.__delattr__('batchnorm')
            self.relu = nn.Identity()

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_channels = int(in_shape[1])  # derive from in_shape for RGB input
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=1),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', self.learning_rate),
            momentum=prm.get('momentum', self.momentum),
            weight_decay=prm.get('weight_decay', 0.0005)
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