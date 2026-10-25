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
        # Fuse conv and BN for faster inference
        if self.conv.bias is not None:
            raise ValueError("fuse should not be called on a layer with bias")
        w_conv = self.conv.weight.float()
        if self.bn.track_running_stats:
            w_bn = torch.diag(self.bn.running_var.rsqrt()) @ torch.diag(self.bn.running_mean) @ w_conv
            b_bn = self.bn.running_mean.float() - self.bn.running_mean.float() @ w_conv @ self.bn.running_var.rsqrt()
        else:
            w_bn = w_conv
            b_bn = self.bn.weight.float()
        fused_conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            self.conv.kernel_size,
            self.conv.stride,
            self.conv.padding,
            bias=True,
        ).requires_grad_(False)
        fused_conv.weight.data = w_bn
        fused_conv.bias.data = b_bn
        fused_conv.out_channels = self.conv.out_channels
        return fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 64, stride=2)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = 32 * 3 * 3 * 64 + 64 * 3 * 3 * 128 + 128 * 3 * 3 * 256 + 256 * 6 * 6 * num_classes
        assert params <= 500000, f"Total parameters: {params}"
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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