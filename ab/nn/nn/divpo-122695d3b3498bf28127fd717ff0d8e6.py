import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused weights to avoid intermediate vars
        fused_weight = self.batchnorm.weight / (self.batchnorm.running_var + self.batchnorm.eps) ** 0.5
        if self.conv.bias is not None:
            fused_bias = self.batchnorm.bias - self.batchnorm.running_mean * self.batchnorm.weight / (self.batchnorm.running_var + self.batchnorm.eps) ** 0.5
        else:
            fused_bias = None
        return nn.Conv2d(self.conv.in_channels, self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, bias=self.conv.bias is not None).requires_grad_(False).to(self.conv.weight.device).float().copy_(self.conv.weight * fused_weight.reshape(1, -1, 1, 1)).float().copy_(torch.tensor(fused_bias) if fused_bias is not None else self.conv.bias)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # Handle channel-first format
        num_classes = int(out_shape[0])

        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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