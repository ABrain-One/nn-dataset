import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def fuse_conv_bn(self):
        # Fuse conv and bn for inference speedup
        fused_conv = (
            self.conv[0]
            .fuse_with(self.conv[1])
            .eval()
            .to(self.conv[0].weight.dtype)
        )

        # Copy learnable parameters
        fused_conv.weight = torch.nn.Parameter(self.conv[0].weight)
        fused_conv.bias = torch.nn.Parameter(self.conv[0].bias)
        return fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(256, 512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(512, 512, stride=1),
            nn.AdaptiveMaxPool2d((6, 6))
        )

        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

        self._initialize_weights()

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
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

parameters = {
    'lr': 0.01,
    'momentum': 0.9,
    'num_classes': 10
}

in_shape = (3, 32, 32)  # Example shape for CIFAR-10 (C, H, W)
out_shape = (10,)  # Example shape for 10 classes
device = torch.device('cpu')  # Example device for CPU

model = Net(in_shape, out_shape, parameters, device)