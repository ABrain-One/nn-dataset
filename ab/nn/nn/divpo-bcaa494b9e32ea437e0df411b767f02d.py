import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Block, 32, 32, 1)
        self.layer2 = self._make_layer(Block, 32, 64, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, block, in_planes, out_planes, blocks, downsample=False):
        layers = []
        for i in range(blocks):
            if i == 0 and downsample:
                layers.append(block(in_planes, out_planes, downsample=True))
            else:
                layers.append(block(out_planes, out_planes, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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
    def __init__(self, in_planes, out_planes, downsample=False):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if residual.shape[1] != out.shape[1]:
            residual = self.conv1(residual)
        out += residual
        out = self.relu(out)
        return out

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_hyperparameter_plan(prm):
    for hp in prm:
        if hp not in supported_hyperparameters():
            return False
    return True

def plan_is_valid(prm):
    return check_hyperparameter_plan(prm)