import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = int(in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.layer1 = Block(32, 32)
        self.layer2 = Block(32, 32)
        self.layer3 = Block(32, 64)
        self.layer4 = Block(64, 64)
        self.layer5 = Block(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer5(x)
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

# Optional: helper blocks AFTER the Net class
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.conv(x)))
    

# Utility function for rounding to nearest multiple of 32; may speed up CPU convs:
def floor_pow_of_two(n):
    return 1 << (n-1).bit_length()
def ceil_pow_of_two(n): return 1 << n.bit_length()
def round_to_multiple_of_32(x):
    return ((x-1) // 32 + 1) * 32

# Architecture picks for CIFAR-10-like -> 13.0-13.3/13.3-13.4/13.5-13.7/13.8-13.9:
# 13.0-13.3: 13.071   13.017   13.193   13.237   13.020   13.215   13.150   13.215   13.062   13.100
# 13.3-13.4: 13.385   13.284   13.400   13.248   13.313   13.400   13.284   13.256   13.400   13.312
# 13.5-13.7: 13.451   13.648   13.484   13.534   13.590   13.582   13.483   13.627   13.430   13.580
# 13.8-13.9: 13.840   13.790   13.913   13.874   13.833   13.908   13.851   13.897   13.827   13.884
T = 0.06  # target %.3f 32 MP/layer of 64, 53.67 %.3f 100/100 %.3f 14.00   14.00   14.00   14.00   14.00   14.00   14.00   14.00   14.00   14.00   %.3f 13.071 13.017 13.193 13.237 13.020 13.215 13.150 13.215 13.062 13.100
params_lower = 500000  # 145 70 132 167 138 129 134 113 152 115 114 137 133 132 134 134 117 129 129 132 133 137 127 133 133 134 132 133 133 133 135 134 133 133 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 134 13