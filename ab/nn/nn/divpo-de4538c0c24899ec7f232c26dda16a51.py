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
        # Fused weights for faster inference
        assert self.training is False
        if self.conv.weight.shape[2:] == (3, 3) and isinstance(self.bn, nn.BatchNorm2d):
            fused_weight = torch.mm(
                self.conv.weight.view(self.conv.weight.shape[0], -1),
                self.bn.weight.view(-1, self.bn.weight.shape[0]))
            self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                                  kernel_size=3, stride=self.conv.stride,
                                  padding=self.conv.padding, bias=True)
            self.conv.weight.data = fused_weight.view(self.conv.weight.shape)
            self.conv.bias.data = self.bn.bias - self.bn.running_mean * self.bn.weight.clone().reciprocal()
            self.bn = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 64, padding=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=2),
            Block(128, 256, stride=2),
            nn.MaxPool2d(2, 2),
            Block(256, 512, stride=2),
            Block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_data):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def export(self):
        conv_layers = []
        for layer in self.features:
            if isinstance(layer, Block):
                if layer.bn is not None:
                    conv_layers.append(layer.conv)
                    conv_layers.append(layer.bn)
                else:
                    conv_layers.append(layer.conv)
        return conv_layers