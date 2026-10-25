import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly deployment
        assert hasattr(self, 'bn') and self.bn.track_running_stats
        w = self.conv.weight.clone().view(self.conv.out_channels, -1)
        b = self.bn.bias.clone() if self.bn.bias is not None else torch.zeros_like(self.bn.running_mean)
        if self.bn.weight is not None:
            w *= self.bn.weight.view(-1, 1)
        if not self.bn.eps == 0:
            w *= (torch.sqrt(self.bn.running_var.view(-1, 1) + self.bn.eps) / torch.sqrt(self.bn.running_var.clone().view(-1, 1) + self.bn.eps))
        if b is not None:
            b *= self.bn.weight
            b = (b - self.bn.running_mean) * w
        self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, bias=True)
        self.conv.weight.data.copy_(w.view(self.conv.weight.shape))
        self.conv.bias.data.copy_(b)
        self.__delattr__('bn')

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self._initialize_weights()

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.weight, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

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
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum', 'weight_decay'}