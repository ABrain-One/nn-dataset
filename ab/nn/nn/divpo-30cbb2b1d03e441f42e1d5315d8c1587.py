import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self.to(device)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = sum(p.numel() for p in self.parameters())
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride if i == 0 else 1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
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

bounds_mb = 4.84 # <= 4.84M params
select_kernel_size = 1 # kernel_size <= 1
class_name = 'Net'

def check_shape(shape):
    return shape.numel() <= 1000000 and shape.prod() <= 3145728

def is_valid_kernel_size(k): return k > 0 and k % 2 == 1

def get_kernel_size(kern_size, in_channels, out_channels):
    return kern_size if is_valid_kernel_size(kern_size) else (in_channels, out_channels)

def get_out_channels(kern_size, in_channels, out_channels):
    return out_channels if is_valid_kernel_size(kern_size) else in_channels

def get_in_channels(kern_size, in_channels, out_channels):
    return in_channels if is_valid_kernel_size(kern_size) else out_channels

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def get_kernel_size(self, in_channels, out_channels):
        return get_kernel_size(self.conv.kernel_size, in_channels, out_channels)

    def get_out_channels(self, in_channels, out_channels):
        return get_out_channels(self.conv.kernel_size, in_channels, out_channels)

    def get_in_channels(self, in_channels, out_channels):
        return get_in_channels(self.conv.kernel_size, in_channels, out_channels)

    def get_output_shape(self, input_shape):
        batch_size, in_channels, height, width = input_shape
        out_channels = self.get_out_channels(in_channels, in_channels)
        kernel_size = self.get_kernel_size(in_channels, out_channels)
        height = (height + 2 * self.conv.padding[0] - kernel_size) // self.conv.stride[0] + 1
        width = (width + 2 * self.conv.padding[1] - kernel_size) // self.conv.stride[1] + 1
        return (batch_size, out_channels, height, width)

    def check_kernel_size(self, kern_size, in_channels, out_channels):
        return is_valid_kernel_size(kern_size) or (in_channels == out_channels)

    def build_conv_layer(self, in_channels, out_channels, kern_size, stride=1, padding=1):
        if not self.check_kernel_size(kern_size, in_channels, out_channels):
            in_channels = self.get_in_channels(kern_size, in_channels, out_channels)
            out_channels = self.get_out_channels(kern_size, in_channels, out_channels)
            kernel_size = self.get_kernel_size(kern_size, in_channels, out_channels)
        return Block(in_channels, out_channels, kernel_size, stride, padding)

    def extra_repr(self):
        return f"in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, " \
               f"kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding}"

    def get_flops(self, input_shape):
        batch_size, in_channels, height, width = input_shape
        out_channels = self.get_out_channels(in_channels, in_channels)
        kernel_size = self.get_kernel_size(in_channels, out_channels)
        multiplications = batch_size * out_channels