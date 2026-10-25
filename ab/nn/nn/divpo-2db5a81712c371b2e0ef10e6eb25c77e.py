import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2, padding=1)
        self.layer3 = Block(64, 128, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.optimizer = None
        self.criteria = None

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

# Auto-generated single-stage model (do not modify)

# ---- timm.models.layers.create_norm ----
def create_norm(norm_type, in_channels, **kwargs):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(in_channels, **kwargs)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, **kwargs)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(in_channels, **kwargs)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm2d(in_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported norm type {norm_type}")

# ---- timm.models.layers.create_act ----
def create_act(act_type, inplace=True):
    if act_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_type == 'hswish':
        return nn.Hardsigmoid(inplace=inplace)
    elif act_type == 'hardswish':
        return nn.Hardswish(inplace=inplace)
    elif act_type == 'mish':
        return nn.Mish(inplace=inplace)
    elif act_type == 'silu':
        return nn.SiLU(inplace=inplace)
    else:
        raise ValueError(f"Unsupported activation type {act_type}")

# ---- Block ----
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type='batchnorm', act_type='relu'):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = create_norm(norm_type, out_channels)
        self.act = create_act(act_type)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# ---- Net (target) ----
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]

        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32, norm_type='batchnorm', act_type='relu')
        self.block2 = Block(32, 64, stride=2, padding=1, norm_type='batchnorm', act_type='relu')
        self.block3 = Block(64, 128, stride=2, padding=1, norm_type='batchnorm', act_type='relu')

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

def supported_hyperparameters():
    return {'lr', 'momentum'}