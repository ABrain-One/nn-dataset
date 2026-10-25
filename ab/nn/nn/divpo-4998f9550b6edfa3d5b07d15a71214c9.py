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
        # Fuse Conv and BN for inference speedup
        if self.conv.in_channels == self.conv.out_channels and self.bn.num_features == self.bn.running_mean.shape[0]:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            ).requires_grad_(False)

            fused_conv.weight.data = self.conv.weight.data
            fused_conv.weight.data *= self.bn.weight.data.view(-1, 1, 1, 1)
            fused_conv.bias.data = self.bn.bias.data - self.bn.running_mean * self.bn.weight.data.view(-1) * self.bn.eps

            self.__class__ = nn.Conv2d
            self.out_channels = self.conv.out_channels
            self._is_nontrivial_resize = self.conv._is_nontrivial_resize
            self._is_permanent_resize = self.conv._is_permanent_resize
            self._is_dequantize = self.conv._is_dequantize
            self._version = self.conv._version
            self._metadata = self.conv._metadata
            self._raw_weight = self.conv._raw_weight
            del self.conv
            del self.bn
            self.weight = fused_conv.weight
            self.bias = fused_conv.bias
        else:
            print("Fusing is only supported when in and out channels are the same and running mean size is 1.")

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Handle channels-first input (C, H, W)
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)

        for param in self.features.parameters():
            param.requires_grad = True

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()