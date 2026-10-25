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
        # Fused weights to reduce params
        self.conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            bias=True
        ).requires_grad_(False)

        w = self.conv.weight.clone().view(self.conv.out_channels, -1)
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        b = self.bn.weight.clone()
        eps = self.bn.eps

        # Calculate new fused weights and bias
        w = w / (torch.norm(w, dim=1, keepdim=True) + eps)
        b = b - w @ (w.t() @ b.t()).t()

        self.conv.weight.data.copy_(w.view(self.conv.weight.shape))
        self.conv.bias.data.copy_(b.view(self.conv.bias.shape))
        self.bn = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

    def fuse(self):
        # Apply weight fusion to blocks for lower inference latency
        for block in self.features.children():
            if isinstance(block, Block):
                block.fuse()
        self.features = nn.Sequential(*list(self.features.children()))
        self._reset_classifier()

    def _reset_classifier(self):
        # Replace Linear with a new one after fusion
        in_features = self.features[-1].conv.out_channels * 6 * 6
        self.classifier = nn.Linear(in_features, self.classifier.out_features).to(self.device)