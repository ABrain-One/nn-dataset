import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, kernel_size, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block1 = nn.Sequential(
            conv_bn(32, 32, 3, 1),
            conv_bn(32, 32, 3, 1)
        )
        self.block2 = nn.Sequential(
            conv_bn(32, 64, 3, 2, groups=32),
            conv_bn(64, 64, 3, 1, groups=64)
        )
        self.block3 = nn.Sequential(
            conv_bn(64, 128, 3, 2, groups=64),
            conv_bn(128, 128, 3, 1, groups=128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def get_flops(self, input_size):
        flops = 0
        input_size = (1, *input_size, 3)
        x = torch.rand(input_size).to(self.device)
        # count parameters
        flops += sum(p.numel() for p in self.parameters())
        # count forward
        flops_count = torch.profiler.flops_model(self, input_size)
        flops += flops_count.total()
        return flops

    def get_params(self):
        return sum(p.numel() for p in self.parameters())
    def get_latency(self, input_size):
        input_size = (1, *input_size, 3)
        with torch.no_grad():
            x = torch.rand(input_size).to(self.device)
            # warmup
            self.forward(x)
            # measure
            latency = torch.jit.trace(self, x).forward(x).mean().item()
        return latency