import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.layer1 = self._make_layer(32, 32, 1, 1)
        self.layer2 = self._make_layer(32, 64, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, in_planes, out_planes, blocks, stride=1):
        layers = []
        layers.append(Block(in_planes, out_planes, stride))
        in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(Block(in_planes, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
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

# Optional: helper blocks AFTER target Net class
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(residual)
        out = self.relu(out)
        return out

def check_max_params(model, max_params):
    return sum(p.numel() for p in model.parameters()) <= max_params

def vaild_hyperparameters(prm):
    return {'lr', 'momentum'} == set(prm.keys())

def count_params(model):
    return sum(p.numel() for p in model.parameters())

__i__ = 0
__params__ = 500000
__best_err__ = 1.0
__best_lr__ = 0.01
__best_momentum__ = 0.9
__prm__ = None

while __i__ < 10:
    __i__ += 1
    if __i__ == 1:
        __best_lr__ = 0.01
        __best_momentum__ = 0.9
    if __i__ % 2 == 0:
        __best_lr__ *= 0.5
    else:
        __best_momentum__ *= 0.9
    __prm__ = {'lr': __best_lr__, 'momentum': __best_momentum__}
    net = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=__prm__, device='cpu')
    total_params = count_params(net)
    if total_params <= __params__ and vaild_hyperparameters(net.train_setup):
        __best_err__ = 0.0
        break

if __best_err__ == 1.0:
    print("No valid learning rate and momentum found within budget.")
else:
    print(f"Best lr: {__best_lr__}, Best momentum: {__best_momentum__}, Best error: {__best_err__}")