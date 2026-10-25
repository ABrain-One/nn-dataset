import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fuse conv and bn for better efficiency
        fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
            bias=True
        ).requires_grad_(False)

        fused_conv.weight.data = self.conv.weight.data
        fused_conv.bias.data = self.bn.weight.data

        def _fuse_bn_into_conv(module, constant_bias=0.0):
            # Move untrained BatchNorm2d constants into Conv2d bias
            module.weight.data = module.weight.data * (module.running_var + module.eps).rsqrt()
            module.weight.data = module.weight.data * module.running_mean
            module.weight.data = module.weight.data + constant_bias
            return module.weight.data

        _fuse_bn_into_conv(fused_conv, self.conv.bias)
        return fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Follow the channels-first (C, H, W) convention
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, num_classes)
        )

        self.to(device)
        self._set_optimizer(prm)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _set_optimizer(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def train_setup(self, prm):
        self.to(self.device)
        self._set_optimizer(prm)
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

parameters = {
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0005
}

in_shape = (3, 32, 32)
out_shape = (10,)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(in_shape, out_shape, parameters, device)