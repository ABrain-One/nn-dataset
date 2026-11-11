import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, in_planes, planes, blocks, downsample=False):
        layers = []
        for i in range(blocks):
            if i == 0 and downsample:
                layers.append(Block(in_planes, planes, stride=2))
            else:
                layers.append(Block(in_planes, planes))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def check_shape(shape):
    assert len(shape) == 4, 'Expected shape of length 4 (B, C, H, W), but got shape' + str(shape)
    assert shape[1] in [1, 3], 'Channels must be 1 or 3, but got shape' + str(shape)

def within_bounds(val, lower, upper, bound_type='lower'):
    if bound_type == 'lower':
        return val.mean().item() >= lower
    elif bound_type == 'upper':
        return val.mean().item() <= upper
    else:
        return lower <= val.mean().item() <= upper

def get_mean_std_from_shape(shape):
    check_shape(shape)
    if shape[1] == 1:
        return 0.5, 0.5
    else:
        return 0.4627, 0.2400

def get_transforms(shape):
    mean, std = get_mean_std_from_shape(shape)
    if within_bounds(shape[2:], 32, 144, bound_type='upper'):
        return nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.RandomResizedCrop(32),
            nn.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            nn.Normalize(mean, std)
        )
    else:
        return nn.Sequential(
            nn.RandomCrop(32, padding=4),
            nn.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            nn.Normalize(mean, std)
        )

def supported_hyperparameters():
    return {'lr', 'momentum'}

def contains_block(model):
    return isinstance(model, Block)

def find_block(models):
    for model in models:
        if contains_block(model):
            return model
    return None

def prune_block(model):
    layer = find_block(model.children())
    if layer is not None:
        layer.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        layer.bn1 = nn.BatchNorm2d(32)
        layer.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        layer.bn2 = nn.BatchNorm2d(32)
        return True
    return False

def prune_model(model, _):
    if prune_block(model):
        return True
    return False

def create_train_transforms(in_shape):
    return get_transforms(in_shape)

def supported_hyperparameters():
    return {'lr', 'momentum'}