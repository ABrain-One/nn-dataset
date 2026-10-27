import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused weights for edge-friendly deployment
        assert self.conv.bias is None
        fused_weight = self.conv.weight.clone()
        fused_weight *= self.bn.weight.reshape(-1, 1, 1, 1)
        return Block(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )._conv_forward(self.conv.weight, fused_weight, self.conv.bias)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Derive from in_shape for edge-friendly deployment
        num_classes = out_shape[0]
        self.features = nn.Sequential(
            Block(in_channels, 64, stride=1, padding=1),
            Block(64, 64, stride=2, padding=1),
            Block(64, 128, stride=1, padding=1),
            Block(128, 128, stride=2, padding=1),
            Block(128, 256, stride=1, padding=1),
            Block(256, 256, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = 500000
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

    def fuse(self):
        # Fuse all blocks for edge-friendly deployment
        fused_features = []
        for module in self.features.children():
            if isinstance(module, Block):
                fused_features.append(module.fuse())
            else:
                fused_features.append(module)
        self.features = nn.Sequential(*fused_features)