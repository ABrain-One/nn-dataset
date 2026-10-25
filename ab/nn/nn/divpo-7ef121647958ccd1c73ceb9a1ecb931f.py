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
        # Fused version of Block, significantly lowers memory consumption and may slightly
        # improve speed when training on GPUs.
        self.conv.fuse_kernel_bias()

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Handle in_shape as (batch, channels, height, width)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 64, stride=2)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = 32 * 9 * 9 + 64 * 16 * 16 + 128 * 32 * 32 + 256 * 6 * 6 + 256 * 3 * 3 + num_classes
        assert params <= 500000, "Total parameter count exceeds limit by " + str(params - 500000)

        for_param_name, for_param_value = "momentum", prm.get("momentum", 0.9)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get("lr", 0.01),
            momentum=for_param_value
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get("lr", 0.01),
            momentum=prm.get("momentum", 0.9)
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()