import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Derive from in_shape to match the shape (batch, channels, height, width)
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __str__(self):
        return "Model\n" + "\n".join(f"{n.type(n):{n.params()}}" for n in self.modules())
    class _Unit(nn.Module):
        def __init__(self, nin, nout):
            super().__init__()
            self.conv = nn.Conv2d(nin, nout, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(nout)
        def forward(self, x):
            return nn.functional.relu(self.bn(self.conv(x)))
        def params(self):
            return self.conv.weight.numel() + self.conv.bias.numel() + self.bn.running_mean.numel() + self.bn.running_var.numel()
        def __repr__(self):
            return f"Unit[conv:{self.conv}, bn:{self.bn}]"
        def type(self, _type):
            self.conv = _type(self.conv)
            self.bn = _type(self.bn)
            return self
    class Block(nn.Module):
        def __init__(self, nin, nout, stride=1):
            super().__init__()
            self.unit = self._unit(nin, nout, stride)
            if stride != 1 or nin != nout:
                self.downsample = self._unit(nin, nout, stride=stride, dilation=2)
            else:
                self.downsample = None
        def _unit(self, nin, nout, stride=1, dilation=1):
            return Net._Unit(nin, nout).type(nn.Conv2d).dilation(dilation).padding(dilation)
        def forward(self, x):
            residual = x if self.downsample is None else self.downsample(x)
            return nn.functional.relu(residual + self.unit(x))
        def params(self):
            return sum(p.numel() for p in self.parameters())
        def __repr__(self):
            return f"Block[unit:{self.unit}, downsample:{self.downsample}]"